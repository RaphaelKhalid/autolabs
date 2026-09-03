export const PROBLEM_CONTEXT = `
Erdos Problem 885. For a positive integer N define D(N) = {|a-b| : a,b positive integers and ab=N}.
The goal k=5 is five distinct positive integers N_1,...,N_5 and five distinct positive differences d_1,...,d_5 such that every d_i belongs to every D(N_j).
Exact cell test: d^2 + 4N must be a perfect square m^2; then a=(m-d)/2, b=(m+d)/2 and ab=N.

Known k=4 certificate (Bremner):
N = [26128575, 291722431, 561117375, 713526975]
d = [126, 16110, 33390, 75390].

Known 2026 search registry from erdosproblemaday.com/day/885-factor-difference-k5:
- Fixed Bremner specializations tested there are bi-maximal; do not repeat them.
- The full elliptic family is not ruled out.
- 71 primitive 4x6 square-additive rectangles are one row short; fixed examples checked were bi-maximal.
- 4.9 billion rational fifth-row candidates with denominator <= 10,000 found none.
- Direct N <= 3,000,000 found none; a structured search with smallest N <= 3,000,000 and other N unbounded found none.
- Difference search with smallest d <= 160 and fifth d <= 50,000 found none.
Do not rerun a covered region. Cite the registry when relying on it. Seek new algebraic families, new parameter regimes, or exact Pareto improvements.

Optimization is exact support. For five proposed differences and candidate N columns, score how many d rows satisfy d^2+4N=m^2 exactly. Sort column supports weakest-first. Goal (5,5,5,5,5); (4,5,5,5,5) beats (4,4,5,5,5). Number size and numerical closeness to a square do not count.
`.trim();
