from mcr.simple_persuasion.basic_bayesian_persuasion import BasicBayesianPersuasion

bp = BasicBayesianPersuasion.simple_binary_example(seed=1)
result = bp.solve(max_iter=1500, step_size=0.1, convergence_tol=1e-7, convergence_patience=40)

print(result["converged"], result["iterations"])
print(result["final_signaling"])
print(result["utility_history"][-5:])
