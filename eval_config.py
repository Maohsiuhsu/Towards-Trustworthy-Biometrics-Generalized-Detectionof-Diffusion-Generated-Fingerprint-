import os

# directory to store the results
results_dir = './results/'
os.makedirs(results_dir, exist_ok=True)

# list of synthesis algorithms
# vals = ['cycleGAN', 'conditional_cycleGAN', 'ProGAN', 'styleGAN', 'styleGAN2-ada', 'DDIM', 'inpaint_with_FK', 'guided',
        # 'LDM', 'styleGAN2']
vals = ['cycleGAN', 'conditional_cycleGAN', 'ProGAN', 'styleGAN', 'styleGAN2-ada', 'DDIM', 'inpaint_with_FK', 'guided',
        'LDM','ControlNet++','StableDiffusion_1.5','StableDiffusion_3.5']        
# vals = ['guided']        
# vals = ['cycleGAN_patch48', 'conditional_cycleGAN_patch48', 'styleGAN2-ada_patch48', 'DDIM_patch48', 'inpaint_with_FK_patch48', 
        # 'guided_patch48', 'LDM_patch48']        
