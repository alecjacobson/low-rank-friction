
# Matrix Completion

Dziugaite & Roy call this "Probabilistic matrix factorization".

Minimize the L2 error at observations using a bilinear reconstruction. Let's say
that the observations are Rᵢⱼ for {ij} ∈ D.

The symmetric version looks like:


minimize ∑ ( Rᵢⱼ - Uᵢᵀ Uⱼ )²
U ∈ ℝⁿˣᵏ

## Discussion

Performance flattens out at k=2. I'm not sure why this model can't 100% over-fit
the given data. It seems if k>n then it should just be able to memorize
everything, so I'd expect to see that happen as k→n

Maybe this is implied by the existence of negative eigenvalues in R (when
treating non-observed entries as zero). These are not easy to get rid of (i.e.,
adding Id removes some but not all of the negative eigenvalues).

Maybe this means we should either have complex values in U or have a diagonal
term that could have negative entries:

minimize ∑ ( Rᵢⱼ - Uᵢᵀ diag(S) Uⱼ )²
U ∈ ℝⁿˣᵏ, S∈ℝᵏ

So the number of unknowns is nk+k. (Any positive value in S could be swallowed
into U via √Sᵢ but this is probably not worth it for small k)

For k=n we can just use eigendecomposition of R (e.g., with explicit zeros).

For smaller k initialized with first k eigen vectors of R, this manages to over fit but only for k≥59. That's 4,425 variables to over
fit to 91 values.

For smaller k initialized randomly, it manages to over fit with k≥3. That's 225
values to overfit 91...


