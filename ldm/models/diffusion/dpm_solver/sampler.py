"""SAMPLING ONLY."""
import torch
import ptp_scripts.ptp_scripts as ptp
import ptp_scripts.ptp_utils as ptp_utils
# from .dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver
from scripts.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

MODEL_TYPES = {
    "eps": "noise",
    "v": "v"
}


class DPMSolverSampler(object):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(model.device)
        self.register_buffer('alphas_cumprod', to_torch(model.alphas_cumprod))

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != self.model.device:
                attr = attr.to(self.model.device)
        setattr(self, name, attr)

    @torch.no_grad()
    def sample(self,
               steps,
               batch_size,
               shape,
               conditioning=None,
               inv_emb=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               t_start=None,
               t_end=None,
               DPMencode=False,
               order=3,
               width=None,
               height=None,
               ref=False,
               top=None, 
               left=None, 
               bottom=None, 
               right=None,
               segmentation_map=None,
               param=None,
               target_height=None, 
               target_width=None,
               center_row_rm=None,
               center_col_rm=None,
               tau_a=0.2,
               tau_b=0.8,
               tau_c=0.2,
               ref_id_list=None,
               top_rr=None,
               bottom_rr=None,
               right_rr=None,
               left_rr=None,
               noise_blend=False,
               frequency_control=None,
               orig_inv_list=None,
               ref_inv_list=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)

        # print(f'Data shape for DPM-Solver sampling is {size}, sampling steps {steps}')

        device = self.model.betas.device
        if x_T is None:
            x = torch.randn(size, device=device)
        else:
            x = x_T

        ns = NoiseScheduleVP('discrete', alphas_cumprod=self.alphas_cumprod)
     
        if DPMencode:
            model_fn = model_wrapper(
                lambda x, t, c, DPMencode, controller, inject, ref_id_list, frequency_control: self.model.apply_model(x, t, c, encode=DPMencode, controller=None, inject=inject, ref_id_list=None, frequency_control=None),
                ns,
                model_type=MODEL_TYPES[self.model.parameterization],
                guidance_type="classifier-free",
                condition=inv_emb,
                unconditional_condition=inv_emb,
                guidance_scale=unconditional_guidance_scale,
            )
            
            noise_list = []

            dpm_solver = DPM_Solver(model_fn, ns)
            data, _ = self.low_order_sample(x, dpm_solver, steps, order, t_start, t_end, device, DPMencode=DPMencode)
            
            for i in range(len(data['model_prev_list'])):
                noise_list.append(data['model_prev_list'][i].clone())


            for step in range(order, steps + 1):
                data = dpm_solver.sample_one_step(data, step, steps, order=order, DPMencode=DPMencode) 

                noise_list.append(data['x'].clone())  
                     
            return data['x'].to(device), noise_list
        else:

            model_fn_cross_decode = model_wrapper(
                lambda x, t, c, DPMencode, controller, inject, ref_id_list, frequency_control: self.model.apply_model(x, t, c, encode=DPMencode, controller=controller, inject=inject, ref_id_list=None, frequency_control=frequency_control),
                ns,
                model_type=MODEL_TYPES[self.model.parameterization],
                guidance_type="classifier-free",
                condition=conditioning,
                unconditional_condition=unconditional_conditioning,
                guidance_scale=unconditional_guidance_scale,
            )

            model_fn_gen = model_wrapper(
                lambda x, t, c, DPMencode, controller, inject, ref_id_list, frequency_control: self.model.apply_model(x, t, c, encode=DPMencode, controller=controller, inject=inject, ref_id_list=ref_id_list, frequency_control=frequency_control),
                ns,
                model_type=MODEL_TYPES[self.model.parameterization],
                guidance_type="classifier-free",
                condition=conditioning,
                unconditional_condition=unconditional_conditioning,
                guidance_scale=unconditional_guidance_scale,
            )
            

            cross_controller = ptp.AttentionStore()
            gen_controller = ptp.AttentionStore()
            Inject_controller = ptp.AttentionStore()
            

            dpm_solver_bg_decode = DPM_Solver(model_fn_cross_decode, ns)
            dpm_solver_gen = DPM_Solver(model_fn_gen, ns)
            
            # decode for cross-attention
            ptp_utils.register_attention_control(self.model, cross_controller, center_row_rm, center_col_rm, target_height, target_width, 
                                                 width, height, segmentation_map=segmentation_map[0, 0].clone(), pseudo_cross=True)
            
            orig_inv_list[0][:, :, param[0]:param[1], param[2]:param[3]] \
                                    = ref_inv_list[0][:, :, top_rr:bottom_rr, left_rr:right_rr].clone() \
                                    * segmentation_map[:, :, top_rr:bottom_rr, left_rr:right_rr] \
                                    + orig_inv_list[0][:, :, param[0]:param[1], param[2]:param[3]].clone() \
                                    * (1 - segmentation_map[:, :, top_rr:bottom_rr, left_rr:right_rr])



            cross, cross_controller = self.low_order_sample(orig_inv_list[0].clone(), dpm_solver_bg_decode, steps, order, t_start, t_end, device, DPMencode=DPMencode,
                                                        controller=cross_controller, ref_init=ref_inv_list[0].clone())
            
            #  generation
         
            Inject_controller = [cross_controller, cross_controller, cross_controller]

            ptp_utils.register_attention_control(self.model, gen_controller, center_row_rm, center_col_rm, target_height, target_width, 
                                                 width, height, segmentation_map=segmentation_map[0, 0].clone())
            gen, _ = self.low_order_sample(x[4], dpm_solver_gen, steps, order, t_start, t_end, device, 
                                           DPMencode=DPMencode, controller=Inject_controller, inject=True, ref_id_list=ref_id_list, noise_blend=noise_blend, frequency_control=frequency_control)


            for i in range(len(gen['model_prev_list'])):
                blended = orig_inv_list[i].clone() 

                # gen['model_prev_list'][i][:, :, param[0] : param[1], param[2] : param[3]] \
                #         = ref_inv_list[i][:, :, top_rr:bottom_rr, left_rr:right_rr].clone() \
                #             * segmentation_map[:, :, top_rr:bottom_rr, left_rr:right_rr] \
                #             + gen['model_prev_list'][i][:, :, param[0]:param[1], param[2]:param[3]] \
                #             * (1 - segmentation_map[:, :, top_rr:bottom_rr, left_rr:right_rr])
                
                blended[:, :, param[0] : param[1], param[2] : param[3]] \
                        = gen['model_prev_list'][i][:, :, param[0] : param[1], param[2] : param[3]].clone()

                gen['model_prev_list'][i] = blended.clone()
            
            del cross_controller, gen_controller, Inject_controller
                        
            cross_controller = ptp.AttentionStore()
            gen_controller = ptp.AttentionStore()
                
           
            for step in range(order, steps + 1):

                if step < int(tau_a*(steps) + 1 - order):
                    inject = True

                    # decode for cross-attention
                    ptp_utils.register_attention_control(self.model, cross_controller, center_row_rm, center_col_rm, target_height, target_width, 
                                                        width, height, segmentation_map=segmentation_map[0, 0].clone(), pseudo_cross=True)
                
                    cross['x'] = orig_inv_list[step].clone()
                    
                    cross['x'][:, :, param[0]:param[1], param[2]:param[3]] \
                                        = ref_inv_list[step][:, :, top_rr:bottom_rr, left_rr:right_rr].clone() \
                                        * segmentation_map[:, :, top_rr:bottom_rr, left_rr:right_rr] \
                                        + cross['x'][:, :, param[0]:param[1], param[2]:param[3]] \
                                        * (1 - segmentation_map[:, :, top_rr:bottom_rr, left_rr:right_rr])

                    ref_inv_list[step][:, :, top_rr:bottom_rr, left_rr:right_rr] \
                                    = gen['x'][:, :, param[0]:param[1], param[2]:param[3]].clone() \
                                    * segmentation_map[:, :, top_rr:bottom_rr, left_rr:right_rr] \
                                    + ref_inv_list[step][:, :, top_rr:bottom_rr, left_rr:right_rr].clone() \
                                    * (1 - segmentation_map[:, :, top_rr:bottom_rr, left_rr:right_rr])

                
                    cross = dpm_solver_bg_decode.sample_one_step(cross, step, steps, order=order, DPMencode=DPMencode, ref_init=ref_inv_list[step].clone(), frequency_control=None)
                else:
                    inject = False
                
                controller = [cross_controller, cross_controller, cross_controller]
            
                # generation
                ptp_utils.register_attention_control(self.model, gen_controller, center_row_rm, center_col_rm, target_height, target_width, width, height, 
                                                    segmentation_map=segmentation_map[0, 0].clone())
                gen = dpm_solver_gen.sample_one_step(gen, step, steps, order=order, DPMencode=DPMencode, controller=controller, inject=inject, ref_id_list=ref_id_list, noise_blend=noise_blend, frequency_control=frequency_control)

                if step < int(tau_c*(steps) + 1 - order):     
                    blended = orig_inv_list[step].clone() 
                    blended[:, :, param[0] : param[1], param[2] : param[3]] \
                            = ref_inv_list[step][:, :, top_rr:bottom_rr, left_rr:right_rr].clone() \
                                * segmentation_map[:, :, top_rr:bottom_rr, left_rr:right_rr] \
                                + blended[:, :, param[0]:param[1], param[2]:param[3]] \
                                * (1 - segmentation_map[:, :, top_rr:bottom_rr, left_rr:right_rr])
                    gen['x'] = blended.clone()

                elif step < int(tau_b*(steps) + 1 - order): 
                    blended = orig_inv_list[step].clone() 
                    
                    
                    blended[:, :, param[0] : param[1], param[2] : param[3]] \
                            = gen['x'][:, :, param[0] : param[1], param[2] : param[3]].clone()
                    
                
                    
                    gen['x'] = blended.clone()
                
            del cross_controller, gen_controller, controller

                

            return gen['x'].to(device), orig_inv_list[-1], ref_inv_list[-1]

            
    
    def low_order_sample(self, x, dpm_solver, steps, order, t_start, t_end, device, DPMencode=False, controller=None, inject=False, ref_init=None, ref_id_list=None, noise_blend=False, frequency_control=None):
        
        t_0 = 1. / dpm_solver.noise_schedule.total_N if t_end is None else t_end
        t_T = dpm_solver.noise_schedule.T if t_start is None else t_start
        
        total_controller = []
        assert steps >= order
        timesteps = dpm_solver.get_time_steps(skip_type="time_uniform", t_T=t_T, t_0=t_0, N=steps, device=device, DPMencode=DPMencode)
        assert timesteps.shape[0] - 1 == steps
        with torch.no_grad():
            vec_t = timesteps[0].expand((x.shape[0]))
            model_prev_list = [dpm_solver.model_fn(x, vec_t, DPMencode=DPMencode, 
                                                   controller=[controller[0][0], controller[1][0], controller[2][0]] if isinstance(controller, list) else controller, 
                                                   inject=inject, ref_init=ref_init, ref_id_list=ref_id_list, noise_blend=noise_blend, frequency_control=frequency_control)]
            
            total_controller.append(controller)
            t_prev_list = [vec_t]
            # Init the first `order` values by lower order multistep DPM-Solver.
            for init_order in range(1, order):
                vec_t = timesteps[init_order].expand(x.shape[0])
                x = dpm_solver.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, vec_t, init_order,
                                                           solver_type='dpmsolver', DPMencode=DPMencode)
                model_prev_list.append(dpm_solver.model_fn(x, vec_t, DPMencode=DPMencode, 
                                                           controller=[controller[0][init_order], controller[1][init_order], controller[2][init_order]] if isinstance(controller, list) else controller,
                                                           inject=inject, ref_init=ref_init, ref_id_list=ref_id_list, noise_blend=noise_blend, frequency_control=frequency_control))
                total_controller.append(controller)
                t_prev_list.append(vec_t)
        
        return {'x': x, 'model_prev_list': model_prev_list, 't_prev_list': t_prev_list, 'timesteps':timesteps}, total_controller
    