"""
    def forward(self, x, indices):

        batch_size, seq_len, hidden_dim = x.shape
        
        # Reshape input and indices for expert routing
        x_reshaped = x.view(-1, hidden_dim)  # [b*t, c]
        indices_flat = indices.view(-1)     # [b*t]
        
        output = torch.zeros_like(x_reshaped)
        
        # Process each expert's tokens
        for i in range(self.num_experts):
            # Get masks for tokens routed to expert i in any of their routes
            mask = indices_flat == i
            if mask.any():
                # Process tokens through expert and accumulate the output
                output[mask] = self.experts[i](x_reshaped[mask])
        
        # Reshape back to original dimensions
        output = output.view(batch_size, seq_len, hidden_dim)
        
        return output, indices
"""