How floating point and integer numbers are represented into the CPU and GPU devices ?



What is Quantization ?
Process of converting large model parameters into fewer bits
It reduces the memory uses and computational power requirements.
Primary aim to replace the high precision floating point numbers(32bits) with the low precision integer numbers(8 bits)
Thus, we can say quantization is a mapping $Q : W \to \tilde{W}$ defined as $$\tilde{W_{ij}} = Q(W_{ij})$$, where W is model weights with high precision and $\tilde{W}$ model wieghts with low precision.

Why do we need quantization ?
Problem : 
1. Deep Neural Networks lie LLaMa 2, GPT 2 etc have more that than 7 billion parameters. Every parameters is a float(32bits) then How much memory space do we needed to save model onto disk ?
$$\frac{7 \times 10^9 \times 32}{8 \times 10^9} = 28GB$$

2. Even saved large model, needed to be loaded onto cache/RAM at the time of inference but standard PC/Smartphone don't have that much storage for RAM. Therefore Inferencing not possible on standard device for larger models.

3. Computing Machines are slow at computation of floating point operations compared to the integer operations. Example Let' $3 \times 8$ be integer operation while $2.21 \times 4.234$ be floating opeation then human and computing devices can easly do the integer operation compared to the floating opeations.

Solution : 
Quantization Aim
1. Reduce the total amound of bits required to represent the each parameters by converting floating point number to the integer numbers. Example : A large model with 32 GB storage can be compressed to the 8/4GB Storage depending upon the types of quantization we used.

Remark : Quantization does't mean truncation or round off floating point numbers.

What are the advantages of Quantization ? 
1. It consumes less memory for storage and loading model to RAM at inference time.
2. It take less inference time due to simpler data types of it's parameters
3. It consume less energy due to less inference time in overall computation.

Quantization : 
Deep Neural Networks are made up of many connected layers of neurons. Example Linear Layer, there are two types of parameter matrices exists, weights and bias commonly represented using floating point numbers.
$$Y = W^T X + B$$

Thus, quantization will map these high precision numbers of both matrices to the low precision numbers while maintaining the accuracy of the model.
$$Q(W, B) = (\tilde{W}, \tilde{B})$$ and quantization is an invertible map therefore $$Q^{-1}(\tilde{W}, \tilde{B}) = (W, B)$$ is called dequantization map.


There are many different types of quantization mapping exists
1. Uniform and Non-uniform Quantization
2. Symmetric and Asymmetric Quantization
3. Dynamic Range Quantization
4. Post-training Quantization
5. Quantization Aware Training


Uniform Quantization 
It is a map $Q : W \to \tilde{W}$ from parameter values to the nearest point into a uniform grid. Example : 8 bit quantization maps a floating point number to 256 discrete levels.

### References
1. https://towardsai.net/p/l/quantization-post-training-quantization-quantization-error-and-quantization-aware-training
2. https://huggingface.co/docs/optimum/concept_guides/quantization
3. https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization
4. https://medium.com/@anhtuan_40207/introduction-to-quantization-09a7fb81f9a4
5. https://apple.github.io/coremltools/docs-guides/source/opt-quantization-algos.html
6. 