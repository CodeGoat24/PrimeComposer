# PrimeComposer: Faster Progressively Combined Diffusion for Image Composition with Attention Steering

Official implementation of [PrimeComposer: Faster Progressively Combined Diffusion for Image Composition with Attention Steering]().

> **PrimeComposer: Faster Progressively Combined Diffusion for Image Composition with Attention Steering**<br>
> Yibin Wang, Weizhong Zhang, Jianwei Zheng, and Cheng Jin <br>
> 
> 
>**Abstract**: <br>
Image composition involves seamlessly integrating given objects into a specific visual context. The current training-free methods rely on composing attention weights from several samplers to guide the generator. However, since these weights are derived from disparate contexts, their combination leads to coherence confusion in synthesis and loss of appearance information. These issues worsen with their excessive focus on background generation, even when unnecessary in this task. This not only slows down inference but also compromises foreground generation quality. Moreover, these methods introduce unwanted artifacts in the transition area. In this paper, we formulate image composition as a subject-based local editing task, solely focusing on foreground generation. At each step, the edited foreground is combined with the noisy background to maintain scene consistency. To address the remaining issues, we propose PrimeComposer, a faster training-free diffuser that composites the images by well-designed attention steering across different noise levels. This steering is predominantly achieved by our Correlation Diffuser, utilizing its self-attention layers at each step. Within these layers, the synthesized subject interacts with both the referenced object and background, capturing intricate details and coherent relationships. This prior information is encoded into the  attention weights, which are then integrated into the self-attention layers of the generator to guide the synthesis process. Besides, we introduce a Region-constrained Cross-Attention to confine the impact of specific subject-related words to desired regions, addressing the unwanted artifacts shown in the prior method thereby further improving the coherence in the transition area. Our method exhibits the fastest inference efficiency and extensive experiments demonstrate our superiority both qualitatively and quantitatively.

![teaser](assets/display.png)

---

</div>

![framework](assets/framework.png)


<!-- ## TODO:
- [ ] Release inference code
- [ ] Release demo
- [ ] Release evaluation code and data 
-->


</div>

<br>



## Additional Results

![sketchy-comp](assets/baseline_compare1.png)

---

</div>


![painting-comp](assets/baseline_compare2.png)

---

</div>


![real-comp](assets/baseline_compare3.png)

---

</div>

