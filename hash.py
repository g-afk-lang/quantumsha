import hashlib
block_data = f"quantum_pow_block_0_timestamp_1735689420"+"1255634993"
print(hashlib.sha256(block_data.encode()).hexdigest())