
import random
import torch
import torch.nn as nn

class Blending(nn.Module):
    def __init__(self, args):
        super(Blending, self).__init__()
        self.args = args
        self.final_weight_range  = (self.args["bl_final_weight_min"], self.args["bl_final_weight_max"])
        self.bl_method_1_op      = torch.Tensor([self.args["bl_method_1_op"]])
        self.bl_method_2_op      = torch.Tensor([self.args["bl_method_2_op"]])

    def forward(self, img_background, img_foreground):
        # bs,c,h,w    = img_background.shape
        self.device = img_background.device

        self.img_background = self.RGB_to_RGBA(img_background)
        self.img_foreground = self.RGB_to_RGBA(img_foreground)

        img_result_1 = self.get_blending_result(method=self.args["bl_method_1"], opacity=self.bl_method_1_op)
        img_result_2 = self.get_blending_result(method=self.args["bl_method_2"], opacity=self.bl_method_2_op)        
        self.weight = torch.Tensor([random.uniform(*self.final_weight_range)]).to(self.device)
        result = img_result_1 * self.weight + img_result_2 * (1 - self.weight)

        return result, self.weight

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"MIB Module Restored from {path}, weight = {self.mib_weight}")

    def RGBA_to_RGB(self, image):
        return image[:,:3,:,:]

    def RGB_to_RGBA(self, image):
        b, c, w, h = image.shape
        img = torch.ones([b, c + 1, w, h]).to(self.device)
        img[:,:3,:,:] = image
        
        return img

    def soft_light(self):
        """
            if A ≤ 0.5: C = (2A-1)(B-B^2) + B
            if A > 0.5: C = (2A-1)(sqrt(B)-B) + B
        """                
        A = self.img_foreground[:, :3, :, :]
        B = self.img_background[:, :3, :, :]
        C = torch.where(A <= 0.5, 
            (2 * A - 1.0)*(B - torch.pow(B,2)) + B,   
            (2 * A - 1.0)*(torch.sqrt(B) - B) + B       
            )
        return C

    def hard_light(self):
        """
            if A ≤ 0.5: C = 2*A*B
            if A > 0.5: C = 1-2*(1-A)(1-B)
        """        
        A = self.img_foreground[:, :3, :, :]
        B = self.img_background[:, :3, :, :]
        C = torch.where(A <= 0.5, 
            2 * A * B,   
            1 - 2 * (1.0 - A)*(1.0 - B)      
            )
        return C

    def lighten(self):
        """
            if B ≤ A: C = A
            if B > A: C = B
        """        
        A = self.img_foreground[:, :3, :, :]
        B = self.img_background[:, :3, :, :]
        C = torch.maximum(A, B)
        return C   

    def darken(self):
        """
            if B ≤ A: C = B
            if B > A: C = A
        """        
        A = self.img_foreground[:, :3, :, :]
        B = self.img_background[:, :3, :, :]
        C = torch.minimum(A, B)
        return C

    def multiply(self):
        """
            C = A * B
        """
        A = self.img_foreground[:, :3, :, :]
        B = self.img_background[:, :3, :, :]
        C = A * B
        return C
    
    def grain_merge(self):
        """
            C = A + B - 0.5
        """
        A = self.img_foreground[:, :3, :, :]
        B = self.img_background[:, :3, :, :]
        C = A + B - 0.5
        return C

    def _compose_alpha(self, opacity):
        comp = self.img_foreground[:,3,:,:]
        
        comp_alpha = comp * opacity
        new_alpha  = comp_alpha + (1.0 - comp_alpha) * self.img_background[:,3,:,:]
        
        ratio = comp_alpha / new_alpha
        ratio[torch.isnan(ratio)] = 0.0
        ratio[torch.isinf(ratio)] = 0.0
        
        return ratio

    def get_blending_result(self, method, opacity):
        opacity = opacity.to(self.device)
        ratio = self._compose_alpha(opacity)
        comp = torch.clip(getattr(self, method)(), 0.0, 1.0)
        ratio_rs = torch.stack([ratio,ratio,ratio],dim=1).to(self.device)
        img_out = comp * ratio_rs + self.img_background[:,:3,:,:] * (1.0 - ratio_rs)
        
        alpha_channel = self.img_background[:,3,:,:]
        alpha_channel = alpha_channel.unsqueeze(dim=1)
        img_out = torch.nan_to_num(torch.cat((img_out, alpha_channel),dim=1))  # add alpha channel and replace nans

        return self.RGBA_to_RGB(img_out).to(self.device)
