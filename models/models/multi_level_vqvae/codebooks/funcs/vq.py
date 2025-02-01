import torch


class VQCodeBookFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x_in: torch.Tensor, embedding_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Autograd function for the index selection. According to the VQ-VAE paper, the gradient for
        x_in should be a mirror, to the x_out.

        Args:
            x_in (torch.Tensor): Input, should be a BS x emb_size x codes (16 x 147)
            embedding_weights (torch.Tensor): Embedding input tensor, should be BS x emb_size x num_codes (4 x 16 x 512)
        """

        ctx.batch_size = x_in.shape[0]

        embedding_batch = embedding_weights.unsqueeze(0).repeat((x_in.shape[0], 1, 1))
        x_in_t = x_in.transpose(1, 2).contiguous().float()
        embedding_batch_t = embedding_batch.transpose(1, 2).contiguous().float()
        embedding_batch_flat = embedding_batch_t.flatten(start_dim=0, end_dim=1)

        distances = torch.cdist(x_in_t, embedding_batch_t)  # 4 x 147 x 512
        indices = torch.argmin(distances, dim=2, keepdim=True)  # 4 x 147 x 1
        x_out = torch.index_select(embedding_batch_flat, dim=0, index=indices.flatten())

        x_out = x_out.view((x_in.shape[0], x_in.shape[2], x_in.shape[1]))
        x_out = x_out.transpose(1, 2).contiguous()

        ctx.save_for_backward(embedding_weights, indices)

        return x_out, indices

    @staticmethod
    def backward(
        ctx, grad_outputs: torch.Tensor, indices: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None]:
        grad_input = None
        grad_emb = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_outputs

        if ctx.needs_input_grad[1]:
            embedding_weights, indices = ctx.saved_variables
            grad_emb = torch.zeros_like(embedding_weights)

            # Feed the gradients into the grad_emb file

            for batch_idx, batch in enumerate(indices.flatten(start_dim=1)):
                running_idx = 0
                for idx in batch:
                    idx_value = idx.item()

                    grad_emb[:, idx_value] += grad_outputs[ # type: ignore
                        batch_idx, :, running_idx
                    ] / (indices.flatten().shape[0])
                    running_idx += 1

        return grad_input, grad_emb, None, None


def vq_code_book_select(x_in: torch.Tensor, emb_batch: torch.Tensor):
    """
    Applies the vq codebook function, allowing to pass gradients through it.
    """
    return VQCodeBookFunc.apply(x_in, emb_batch)
