import torch
from torch.distributions import Independent, Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform
import numpy as np

batch_size = 400
torch.set_default_dtype(torch.float64)
n = 40
print(n)
done = False
i = 0
while not done:
    mu = torch.as_tensor(np.random.random([batch_size, n]))
    log_std = torch.as_tensor(np.random.random([batch_size, n]))
    transform = TransformedDistribution(Independent(Normal(mu, log_std.exp()), 1), TanhTransform())
    input = transform.rsample()
    output = transform.log_prob(input)
    if torch.isnan(output).any().item():
        done = True
    if (input == -1).any() or (input == 1).any():
        print("somethings wrong...")
print(output)
print("something was wrong")
