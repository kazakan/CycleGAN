import torch
import torch.nn as nn


class GANLoss:
    def __init__(self, real_target_value=1.0, fake_target_value=0.0) -> None:
        super().__init__()
        self.real_target_value = real_target_value
        self.fake_target_value = fake_target_value

        self.lossfunc = nn.MSELoss()

    def __call__(self, prediction, is_target_real):
        target = torch.full_like(
            prediction,
            self.real_target_value if is_target_real else self.fake_target_value,
        )
        loss = self.lossfunc(prediction, target)
        return loss


class CycleGANGeneratorLoss:
    def __init__(self, lambda1, lambda2, lambda_I) -> None:
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda_I = lambda_I

        self.identityloss_func = nn.L1Loss()
        self.ganloss_func = GANLoss()
        self.cycleloss_func = nn.L1Loss()

    def __call__(
        self, real_A, real_B, fake_A, fake_B, rec_A, rec_B, G_A, G_B, D_A, D_B
    ) -> torch.Tensor:
        # identity loss
        if self.lambda_I > 0:
            loss_idt_A = self.identityloss_func(G_A(real_B), real_B) * self.lambda2
            loss_idt_B = self.identityloss_func(G_B(real_A), real_A) * self.lambda1
        else:
            loss_idt_A = 0
            loss_idt_B = 0

        # Gan loss
        loss_G_A = self.ganloss_func(D_A(fake_B), True)
        loss_G_b = self.ganloss_func(D_B(fake_A), True)

        # Cycleloss
        loss_cycle_A = self.cycleloss_func(rec_A, real_A) * self.lambda1
        loss_cycle_B = self.cycleloss_func(rec_B, real_B) * self.lambda2

        return (
            loss_G_A + loss_G_b + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
        )


class CycleGANDiscriminatorLoss:
    def __init__(self) -> None:
        self.ganloss_func = GANLoss()

    def __call__(self, D, real, fake) -> torch.Tensor:
        yhat = D(real)
        loss_real = self.ganloss_func(yhat, True)

        yhat = D(fake.detach())
        loss_fake = self.ganloss_func(yhat, False)
        return (loss_real + loss_fake) * 0.5
