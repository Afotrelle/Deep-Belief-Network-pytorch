from RBM import RBM
import torch

class GaussianBernoulliRBM(RBM):

    '''
    Visible layer can assume real values.
    Hidden layer assumes Binary values only.

    '''

    def to_visible(self, X):
        '''
        The visible units follow Gaussian distributions here.

        :param X: torch tensor shape = (n_samples , n_features)
        :returns: X_prob - the new reconstructed layers (means)
                  sample_X_prob - sample of new layer (Gibbs Sampling)

        '''

        X_prob = torch.matmul(X, self.W.transpose(0, 1))
        X_prob = torch.add(X_prob, self.v_bias)

        # Noise is placed on the same device as the model
        sample_X_prob = X_prob + torch.randn_like(X_prob)

        return X_prob, sample_X_prob
