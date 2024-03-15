from typing import List

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class SophiaG(Optimizer):
    """
    SophiaG optimizer class is an optimizer that uses hessians to
    adaptively update the learning rate per parameter.

    Args
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-4)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.99))
        rho (float, optional): rho parameter (default: 0.04)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.0)
        maximize (bool, optional): maximize the loss function (default: False)
        capturable (bool, optional): capture the batch size (default: False)
        dynamic (bool, optional): use dynamic learning rate (default: False)


    >>> optimizer = SophiaG(model.parameters(), lr=5e-5, betas=(0.9, 0.999), weight_decay=0.01)
    >>> lr_scheduler = get_lr_scheduler_with_warmup(optimizer, "linear", num_warmup_steps=500, max_train_steps=training_args.max_steps)



    """

    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.965, 0.99),
        rho=0.04,
        weight_decay=1e-1,
        *,
        maximize: bool = False,
        capturable: bool = False,
        dynamic: bool = False,
    ):
        """
        Initialize the optimizer.
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= rho:
            raise ValueError(f"Invalid rho parameter at index 1: {rho}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(
            lr=lr,
            betas=betas,
            rho=rho,
            weight_decay=weight_decay,
            maximize=maximize,
            capturable=capturable,
            dynamic=dynamic,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        """
        Set the state of the optimizer.
        """
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("capturable", False)
            group.setdefault("dynamic", False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["step"]
        )
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]))

    @torch.no_grad()
    def update_hessian(self):
        """
        Update the hessian.
        """
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = (
                        torch.zeros((1,), dtype=torch.float, device=p.device)
                        if self.defaults["capturable"]
                        else torch.tensor(0.0)
                    )
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["hessian"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                if "hessian" not in state.keys():
                    state["hessian"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                state["hessian"].mul_(beta2).addcmul_(
                    p.grad, p.grad, value=1 - beta2
                )

    @torch.no_grad()
    def update_exp_avg(self):
        """
        Update the exponential average.
        """
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                state["exp_avg"].mul_(beta1).add_(p.grad, alpha=1 - beta1)

    @torch.no_grad()
    def step(self, closure=None, bs=5120):
        """
        Perform a step of the optimizer.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.update_hessian()
        self.update_exp_avg()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            state_steps = []
            hessian = []
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                params_with_grad.append(p)

                if p.grad.is_sparse:
                    raise RuntimeError("Hero does not support sparse gradients")
                grads.append(p.grad)
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = (
                        torch.zeros((1,), dtype=torch.float, device=p.device)
                        if self.defaults["capturable"]
                        else torch.tensor(0.0)
                    )
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["hessian"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                if "hessian" not in state.keys():
                    state["hessian"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avgs.append(state["exp_avg"])
                state_steps.append(state["step"])
                hessian.append(state["hessian"])

                if self.defaults["capturable"]:
                    bs = (
                        torch.ones((1,), dtype=torch.float, device=p.device)
                        * bs
                    )

            self._sophiag(
                params_with_grad,
                grads,
                exp_avgs,
                hessian,
                state_steps,
                bs=bs,
                beta1=beta1,
                beta2=beta2,
                rho=group["rho"],
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                maximize=group["maximize"],
                capturable=group["capturable"],
            )

        return loss

    def _sophiag(
        self,
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        hessian: List[Tensor],
        state_steps: List[Tensor],
        capturable: bool = False,
        *,
        bs: int,
        beta1: float,
        beta2: float,
        rho: float,
        lr: float,
        weight_decay: float,
        maximize: bool,
    ):
        """
        SophiaG function.
        """
        if not all(isinstance(t, torch.Tensor) for t in state_steps):
            raise RuntimeError(
                "API has changed, `state_steps` argument must contain a list of"
                " singleton tensors"
            )

        self._single_tensor_sophiag(
            params,
            grads,
            exp_avgs,
            hessian,
            state_steps,
            bs=bs,
            beta1=beta1,
            beta2=beta2,
            rho=rho,
            lr=lr,
            weight_decay=weight_decay,
            maximize=maximize,
            capturable=capturable,
        )

    def _single_tensor_sophiag(
        self,
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        hessian: List[Tensor],
        state_steps: List[Tensor],
        *,
        bs: int,
        beta1: float,
        beta2: float,
        rho: float,
        lr: float,
        weight_decay: float,
        maximize: bool,
        capturable: bool,
    ):
        """
        SophiaG function for single tensor.
        """
        for i, param in enumerate(params):
            grad = grads[i] if not maximize else -grads[i]
            exp_avg = exp_avgs[i]
            hess = hessian[i]
            step_t = state_steps[i]

            if capturable:
                assert param.is_cuda
                assert step_t.is_cuda
                assert bs.is_cuda

            if torch.is_complex(param):
                grad = torch.view_as_real(grad)
                exp_avg = torch.view_as_real(exp_avg)
                hess = torch.view_as_real(hess)
                param = torch.view_as_real(param)

            # update step
            step_t += 1

            # Perform stepweight decay
            param.mul_(1 - lr * weight_decay)

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

            if capturable:
                step_size = lr
                step_size_neg = step_size.neg()

                ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(
                    None, 1
                )
                param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)
            else:
                step_t.item()
                step_size_neg = -lr

                ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(
                    None, 1
                )
                param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)
