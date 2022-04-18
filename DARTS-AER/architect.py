""" Architect controls architecture of cell by computing gradients of alphas """
import copy
import torch
import genotypes as gt


class Architect():
    """ Compute gradients of alphas """
    def __init__(self, net, w_momentum, w_weight_decay):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net = net
        self.v_net = copy.deepcopy(net)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay

    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, xi, w_optim, uniform_forward=False, alpha_epsilon=None, feature_epsilon=None, entropy_reg=None):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # do virtual step (calc w`)
        self.virtual_step(trn_X, trn_y, xi, w_optim, uniform_forward, alpha_epsilon, feature_epsilon)

        # calc unrolled loss
        loss = self.v_net.loss(val_X, val_y, entropy_reg=entropy_reg) # L_val(w`)

        # compute gradient
        v_alphas = tuple(self.v_net.alphas())
        v_weights = tuple(self.v_net.weights())
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights)
        dalpha = v_grads[:len(v_alphas)]
        dw = v_grads[len(v_alphas):]

        hessian = self.compute_hessian(dw, trn_X, trn_y)

        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():
            for alpha, da, h in zip(self.net.alphas(), dalpha, hessian):
                if isinstance(xi, list):
                    alpha.grad = da - xi[-1]*h
                else:
                    alpha.grad = da - xi*h

    def virtual_step(self, trn_X, trn_y, xi, w_optim, uniform_forward=False, alpha_epsilon=None, feature_epsilon=None):
        """
        Compute unrolled weight w' (virtual step)

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        # forward & calc loss
        loss = self.net.loss(trn_X, trn_y, uniform_forward, alpha_epsilon, feature_epsilon) # L_trn(w)

        # compute gradient
        if isinstance(w_optim, dict) and isinstance(xi, list):
            gradients = []
            for i, op_name in enumerate(gt.PRIMITIVES+['others']):
                gradients.append(torch.autograd.grad(loss, self.net.op_weights(op_name), retain_graph=False if op_name == 'others' else True) if self.net.op_weights(op_name) is not None else None)
        else:
            gradients = torch.autograd.grad(loss, self.net.weights())

        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            if isinstance(w_optim, dict) and isinstance(xi, list):
                for i, op_name in enumerate(gt.PRIMITIVES+['others']):
                    if w_optim[op_name] is not None:
                        assert self.net.op_weights(op_name) is not None and self.v_net.op_weights(op_name) is not None and gradients[i] is not None
                        for w, vw, g in zip(self.net.op_weights(op_name), self.v_net.op_weights(op_name), gradients[i]):
                            m = w_optim[op_name].state[w].get('momentum_buffer', 0.) * self.w_momentum
                            vw.copy_(w - xi[i] * (m + g + self.w_weight_decay*w))
            else:
                for w, vw, g in zip(self.net.weights(), self.v_net.weights(), gradients):
                    m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                    vw.copy_(w - xi * (m + g + self.w_weight_decay*w))

            # synchronize alphas
            for a, va in zip(self.net.alphas(), self.v_net.alphas()):
                va.copy_(a)

    def compute_hessian(self, dw, trn_X, trn_y):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d
        loss = self.net.loss(trn_X, trn_y)
        dalpha_pos = torch.autograd.grad(loss, self.net.alphas()) # dalpha { L_trn(w+) }

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p -= 2. * eps * d
        loss = self.net.loss(trn_X, trn_y)
        dalpha_neg = torch.autograd.grad(loss, self.net.alphas()) # dalpha { L_trn(w-) }

        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d

        hessian = [(p-n) / (2.*eps) for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian
