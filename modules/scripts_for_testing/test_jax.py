import jax.numpy as jnp
import jax
from jax import jit, partial


@jit
def df_lt(a, b, temp):
	# if a > b
	return 1./ (1. + jnp.exp((a - b) / temp))

members = jnp.transpose(jnp.array([[1., 2., 3., 5., 6., 7., 9., 10.], [2., 3., 4., 7., 9., 10., 11., 15.]]))

print(members.shape)

errs = members[1:, :] - members[:-1, :]
thres = 1.5

split_ = jnp.round(df_lt(errs, thres, 0.01)).astype(jnp.bool_)

print(split_)
print(errs)


# def calculate(difs, splits):
# 	buf = 0.
# 	cnt = 0.
# 	local_cnt = 0.
# 	t_l = []

# 	for idx, err in enumerate(difs):
# 		buf += err
# 		cnt += 1.
# 		local_cnt += 1.
# 		split = splits[idx]
# 		if split:
# 			t_l.append(buf / cnt)
# 			buf = 0.
# 			local_cnt = 0.

# 	t_l.append(buf / cnt)

# 	return jnp.stack(t_l)

# a = jax.vmap(calculate, (1, 1), 1)(errs, split_)

# print(a)

# nz = jnp.nonzero(split_[:,0])[0]
# print(nz)
# r = jnp.vstack([jnp.concatenate([jnp.cumsum(split_[nz[i]:nz[i+1],0]), jnp.zeros(7-nz[i+1]+nz[i])]) for i in range(len(nz))])
# print(r)
# exit()

def calculate(splits):
	print(splits)
	nz = jnp.nonzero(splits)[0]
	return jnp.sum([jnp.concatenate([jnp.cumsum(split_[nz[i]:nz[i+1],0]), jnp.zeros(7-nz[i+1]+nz[i])]) for i in range(len(nz))])

a = jax.vmap(calculate, (1,), 1)(split_)

print(a)