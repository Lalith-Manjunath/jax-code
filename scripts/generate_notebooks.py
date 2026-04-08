import os
import json

def create_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def markdown_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" if i < len(text.split("\n")) - 1 else line for i, line in enumerate(text.split("\n"))]
    }

def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" if i < len(code.split("\n")) - 1 else line for i, line in enumerate(code.split("\n"))]
    }

notebooks_dir = "d:/Code/jax-code/notebooks"
os.makedirs(notebooks_dir, exist_ok=True)

# ------------------------------------------------------------------------------------------------------
# Notebook 1: JAX Basics & The SimplyModule Functional Design
# ------------------------------------------------------------------------------------------------------
nb1_cells = [
    markdown_cell("""# 01: JAX Foundations & The Functional Software Pattern
Welcome to the first notebook in your journey to learning JAX, Large Language Model Architectures, and the `simply` codebase!

If your goal is to use `simply` for your own LLM experiments, you must first transition from an **Object-Oriented** mindset (common in PyTorch) to a **Functional** mindset (mandatory in JAX). Let's dive in.
"""),
    code_cell("""# Setup environment for imports (Make sure you are running from jax-code root or have simply in your PYTHONPATH)
import sys
import os
sys.path.append(os.path.abspath('../third-party/simply'))

import jax
import jax.numpy as jnp
"""),
    markdown_cell("""## 1. Pure Functions vs Mutating State
In an OOP framework like PyTorch, a layer often mutates its internal state when running a forward pass, or when gradients are applied. 
`self.weight.grad = ...`

JAX requires **Pure Functions**. A pure function:
1. Cannot mutate its inputs or global state.
2. Given identical inputs, always produces identical outputs.

Because of this, neural networks in JAX (and in `simply`) are explicitly passed their parameters on every single forward pass.
"""),
    markdown_cell("""## 2. JAX JIT (Just-In-Time Compilation)
JIT compiles your python functions down to XLA (Accelerated Linear Algebra) code that runs incredibly fast on CPU, GPU, or TPU.
"""),
    code_cell("""def relu(x):
    return jnp.maximum(0, x)

# JIT compile the function
fast_relu = jax.jit(relu)

x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
print("Standard ReLU output:", relu(x))
print("JIT ReLU output:", fast_relu(x))
"""),
    markdown_cell("""## 3. The `SimplyModule` Abstraction
In `simply/utils/module.py`, you'll find `SimplyModule`, which acts as a structured way to separate the *definition* of a module from its *state*. 
It has two core methods:
- `init(prng_key)`: Generates and returns a dictionary of randomly initialized parameters.
- `apply(params, x)`: Computes the forward pass using explicitly passed parameters.

This strictly conforms to JAX's pure function requirement!
"""),
    code_cell("""from simply.utils.module import SimplyModule, ModuleRegistry
from dataclasses import dataclass

@ModuleRegistry.register
@dataclass
class MySimpleLinear(SimplyModule):
    features: int

    def init(self, prng_key):
        # Return parameters for this module
        return {"weight": jax.random.normal(prng_key, (10, self.features))}
        
    def apply(self, params, x):
        # Notice how `params` is explicitly passed here! 
        # State is NOT stored on `self`.
        return jnp.dot(x, params["weight"])
"""),
    code_cell("""# Let's test our simple module!
rng = jax.random.PRNGKey(42)
module = MySimpleLinear(features=5)

# 1. Initialize parameters
params = module.init(rng)
print("Initialized parameters keys:", params.keys())
print("Weight shape:", params["weight"].shape)

# 2. Run the forward pass
x_input = jnp.ones((2, 10))
output = module.apply(params, x_input)
print("Output shape:", output.shape)
"""),
    markdown_cell("""## Exercise: Fill-in-the-blank
Create an instance of `MySimpleLinear`, generate the parameters, and `vmap` (vectorizing map) the `apply` function over a newly shaped input of `(5, 2, 10)` representing `[batch, sequence, dim]`.

Remember `jax.vmap` automatically vectorizes pure functions over leading axes.
"""),
    code_cell("""# 1. Instantiate module
# my_module = ???

# 2. Vectorized apply function using jax.vmap
# We sweep over the batch dimension of x, but the parameters remain the same!
# vectorized_apply = jax.vmap(my_module.apply, in_axes=(???, ???))

# test_input = jnp.ones((5, 2, 10))
# out = vectorized_apply(params, test_input)
# print(out.shape) # Should be (5, 2, 5)
"""),
    markdown_cell("""### Solution"""),
    code_cell("""my_module = MySimpleLinear(features=5)

# in_axes specifies which axes of the arguments to map over.
# params dict is shared across the batch (None), x sweeps over dimension 0 (0)
vectorized_apply = jax.vmap(my_module.apply, in_axes=(None, 0))

test_input = jnp.ones((5, 2, 10))
# The 'params' variable was created in the cell above
out = vectorized_apply(params, test_input)
print("Vectorized output shape:", out.shape)
""")
]

with open(os.path.join(notebooks_dir, "01_jax_basics_and_design.ipynb"), "w") as f:
    json.dump(create_notebook(nb1_cells), f, indent=2)

# ------------------------------------------------------------------------------------------------------
# Notebook 2: Model Layers & Einsum
# ------------------------------------------------------------------------------------------------------
nb2_cells = [
    markdown_cell("""# 02: Model Layers & Einsum in Simply
Now that you understand pure functional modules, let's explore how `simply` constructs the atomic layers of LLMs, such as `LayerNorm` and linear layers.

To build large language models effectively, `simply` relies heavily on `einsum` (Einstein Summation) notation instead of rigid operations like `np.matmul` or PyTorch's `nn.Linear`.
"""),
    code_cell("""import sys
import os
sys.path.append(os.path.abspath('../third-party/simply'))

import jax
import jax.numpy as jnp
"""),
    markdown_cell("""## 1. The magic of JNP.EINSUM
`einsum` allows you to express complex tensor contractions using elegant index strings. Let's see how `simply` uses it to simplify attention projections.
"""),
    code_cell("""# A standard batched matrix multiplication using matmul:
# Context: We have inputs [batch, seq, dim_in] and a weight matrix [dim_in, dim_out]
batch_size, seq_len, dim_in, dim_out = 2, 4, 8, 16
x = jnp.ones((batch_size, seq_len, dim_in))
w = jax.random.normal(jax.random.PRNGKey(0), (dim_in, dim_out))

# The traditional way (dot product handling axes manually)
out_traditional = jnp.dot(x, w)

# The EINSUM way (b=batch, s=seq, i=dim_in, o=dim_out)
# "bsi,io->bso"
# Contract the 'i' dimension. Keep 'b', 's', and 'o'.
out_einsum = jnp.einsum('bsi,io->bso', x, w)

print("Are they almost equal?", jnp.allclose(out_traditional, out_einsum))
"""),
    markdown_cell("""## 2. Using simply's EinsumLinear
Because `einsum` is so powerful, `simply/utils/module.py` ships with `EinsumLinear`. This is the workhorse of their Transformer implementation!

Let's read `EinsumLinear` code or instantiate it directly.
"""),
    code_cell("""from simply.utils.module import EinsumLinear

# Instead of passing 'in_features' and 'out_features', we pass an equation and shape!
# This allows incredibly flexible projection logic (e.g. mapping directly into heads).
linear = EinsumLinear(
    eqn='bsi,io->bso', 
    weight_shape=(8, 16), 
    bias_term='o' # Add bias to the output dimension 'o'
)

rng = jax.random.PRNGKey(1)
params = linear.init(rng)

output = linear.apply(params, x)
print("EinsumLinear output shape:", output.shape)
"""),
    markdown_cell("""## Exercise: Projecting into Multiple Heads
In multi-head attention, we project an input `x` (shape `[batch, seq, model_dim]`) into queries `q` with shape `[batch, seq, num_heads, head_dim]`.

Write the `eqn` for an `EinsumLinear` to cleanly accomplish this projection! Use the characters `b` (batch), `s` (seq), `m` (model_dim), `h` (num_heads), `d` (head_dim).
"""),
    code_cell("""# Write your solution here:
# eqn = '???'
# weight_shape = (???, ???, ???)

# q_proj = EinsumLinear(
#    eqn=eqn,
#    weight_shape=weight_shape
# )
"""),
    markdown_cell("""### Solution:"""),
    code_cell("""num_heads, head_dim, model_dim = 4, 64, 256
x = jnp.ones((2, 10, model_dim))

# Equation:
# Input: b s m
# Weight: m h d
# Output: b s h d (Contract m)
q_proj = EinsumLinear(
    eqn='bsm,mhd->bshd',
    weight_shape=(model_dim, num_heads, head_dim)
)

params = q_proj.init(jax.random.PRNGKey(0))
q = q_proj.apply(params, x)
print("Query shape:", q.shape) # Should be (2, 10, 4, 64)
""")
]

with open(os.path.join(notebooks_dir, "02_model_layers_and_einsum.ipynb"), "w") as f:
    json.dump(create_notebook(nb2_cells), f, indent=2)

# ------------------------------------------------------------------------------------------------------
# Notebook 3: Attention & KV Cache
# ------------------------------------------------------------------------------------------------------
nb3_cells = [
    markdown_cell("""# 03: Attention Mechanisms & Autoregressive KV Cache
This notebook connects the layers to build full attention logic. During token generation (inference), LLMs suffer from recomputation overhead. `simply/model_lib.py` solves this functionally using a Key/Value cache.
"""),
    code_cell("""import sys
import os
sys.path.append(os.path.abspath('../third-party/simply'))

import jax
import jax.numpy as jnp
"""),
    markdown_cell("""## 1. Causal Masking
Before a Transformer can generate the next token, it must ensure that attention can only look 'backwards'. Future tokens cannot influence past ones.
Let's see the heart of the attention loop.
"""),
    code_cell("""# Simulate sequence positions for Queries and KV pairs
# Shape: [batch_size=1, seq_len=4]
q_positions = jnp.array([[0, 1, 2, 3]])
kv_positions = jnp.array([[0, 1, 2, 3]])

# Expand to create causal mask of shape [batch, q_seq, kv_seq]
# 'bl -> bl1' expanding queries, 'bl -> b1l' expanding keys
q_exp = jnp.expand_dims(q_positions, axis=-1)
kv_exp = jnp.expand_dims(kv_positions, axis=-2)

causal_mask = q_exp >= kv_exp
print("Causal Matrix:")
# True means unmasked (can attend), False means masked.
print(causal_mask[0].astype(int))
"""),
    markdown_cell("""## 2. Functional KV Caching
In `simply/model_lib.py`, you'll see a function called `updated_decode_state`. Let's understand why it exists.

When generating a token autoregressively, the sequence length is `seq=1`, but the attention must attend to all previous K and V matrices.
Unlike an object that mutates a `.cache` attribute internally, this function takes the raw dictionary `decode_state` and returns an updated *new* dictionary with the new token dynamically inserted at the `cache_position`.
"""),
    code_cell("""from simply.model_lib import updated_decode_state

# Let's mock a fast functional decoding step!
decode_state = {
    'k': jnp.zeros((1, 8, 4, 64)), # [batch, max_seq_len, num_heads, head_dim]
    'v': jnp.zeros((1, 8, 4, 64)),
    'segment_positions': jnp.zeros((1, 8), dtype=jnp.int32),
    'segment_ids': jnp.zeros((1, 8), dtype=jnp.int32),
}

# The new token's key and value:
curr_k = jnp.ones((1, 1, 4, 64))
curr_v = jnp.ones((1, 1, 4, 64))
segment_position = jnp.array([[2]]) # Pretend we are decoding position index 2
segment_ids = jnp.array([[0]])

# Get the updated cache functionally
new_k, new_v, _, _, next_decode_state = updated_decode_state(
    curr_k, curr_v, segment_position, segment_ids, decode_state, window_size=0
)

# Observe the dynamic update! The `k` array has 1s at index 2 now.
print("KV cache at position 1 (empty):", next_decode_state['k'][0, 1, 0, 0])
print("KV cache at position 2 (updated):", next_decode_state['k'][0, 2, 0, 0])
"""),
    markdown_cell("""## Exercise:
Inspect the functional attention block `attn` in `simply/model_lib.py`. How does it apply the attention soft cap? Use its implementation to build a dummy attention wrapper.
""")
]

with open(os.path.join(notebooks_dir, "03_attention_and_kv_cache.ipynb"), "w") as f:
    json.dump(create_notebook(nb3_cells), f, indent=2)

# ------------------------------------------------------------------------------------------------------
# Notebook 4: Advanced Architectures (MoE)
# ------------------------------------------------------------------------------------------------------
nb4_cells = [
    markdown_cell("""# 04: Advanced Architectures - Mixture of Experts (MoE)
As language models grow larger, dense MLPs become extremely expensive. `simply` provides `MoEFeedForward` in `model_lib.py`.

In a Mixture of Experts, a small router network decides which 'experts' (smaller MLPs) process a particular token.
"""),
    code_cell("""import sys
import os
sys.path.append(os.path.abspath('../third-party/simply'))

import jax
import jax.numpy as jnp
"""),
    markdown_cell("""## 1. Top-K Routing Logic
The core component of MoE is computing routing probabilities and selecting the Top-K experts. Let's write the routing equation step-by-step just like in `MoEFeedForward`.
"""),
    code_cell("""batch_size, seq_len = 2, 5
num_experts = 8
num_experts_per_token = 2

# Assume the router outputs logits for each expert per token:
router_logits = jax.random.normal(jax.random.PRNGKey(42), (batch_size, seq_len, num_experts))

# Step 1: Get the Top-K logits and their indices
# selected_router_logits shape: [batch, seq, top_k]
selected_router_logits, selected_indices = jax.lax.top_k(router_logits, k=num_experts_per_token)

# Step 2: Normalize the probabilities over only the chosen experts
routing_weights = jax.nn.softmax(selected_router_logits, axis=-1)

print(f"Token (batch 0, pos 0) assigned to experts: {selected_indices[0,0]}")
print(f"With routing weights: {routing_weights[0,0]}")
"""),
    markdown_cell("""## Why is this functional implementation fast in JAX?
When JAX compiles `jax.lax.top_k` and the subsequent scattered sparse dot-products (`megablox` or `ragged_dot` as defined in `model_lib.py`), the XLA compiler optimizes the parallel dispatch across experts over TPUs/GPUs.

You can instantiate a full MoE layer in `simply` for your experiments directly:
"""),
    code_cell("""from simply.model_lib import MoEFeedForward
from unittest.mock import Mock

# Creating a mock sharding config (since we are testing locally without multiple TPUs)
class DummyConfig:
    activation_partition = None
    ffn0_partition = None
    ffn1_partition = None

moe_layer = MoEFeedForward(
    model_dim=128,
    expand_factor=4,
    sharding_config=DummyConfig(),
    num_experts=4,
    num_experts_per_token=1
)

params = moe_layer.init(jax.random.PRNGKey(0))
x = jnp.ones((2, 5, 128))
inputs_mask = jnp.ones((2, 5), dtype=bool)

# Notice we explicitly pass the params and the mask
outputs, extra = moe_layer.apply(params, x, inputs_mask=inputs_mask)
print("MoE Output shape:", outputs.shape)
""")
]

with open(os.path.join(notebooks_dir, "04_advanced_architectures.ipynb"), "w") as f:
    json.dump(create_notebook(nb4_cells), f, indent=2)

print("Successfully generated all notebooks!")
