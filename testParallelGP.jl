#@everywhere using LsqFit
if nprocs() == 1
    addprocs(19)
end

@everywhere include("GeneticProgramming.jl")

# define input data
@everywhere X = collect(linspace(0,2*pi,50));
# define output data that we will attempt to reproduce.
# Since Y is used in the fitness funct, not defined in
# the library, we need to initialize Y on all procs.
@everywhere Y = (X-1).*sin(2*X);

# create a library of nodes that we can use to build
# syntax trees. Deciding which functions and terminals
# to include is problem dependent.
library = gpLibrary()
for i in -10:10
    library.addTerminal(i)
end
# add input variable as a terminal.
library.addTerminal(X)
library.addFunction(+,2)
library.addFunction(-,2)
library.addFunction(*,2)
library.addFunction(/,2)
library.addTerminal(pi)
library.addFunction(sin,1)
library.addFunction(cos,1)

# define a fitness function. In this case, we will
# use RMSE. 
fitFunct = function(treeOutput)
    sqrt(mean((treeOutput-Y).^2))
end

# Initialize the GP to run in parallel. The vars dictionary defines 
# a map that assigns prettier printing to variables that would
# otherwise require many characters to print, such as printing
# a vector or pi.
println("Creating GP.")
G = GP(library,fitFunct,vars=Dict("X"=>X,"pi"=>pi),islandCount=nprocs(),mutate=.5,popSize=100);
println("End creating GP.")
# Run the GP in serial for some desired number of genertations.
println("Begin Run.")
G.run(10000);
println("End Run.")

# Plot the Pareto front, which compares the number of nodes in a 
# tree to the fitness of the tree. We want trees with low fitness
# and low number of nodes. We think about the number of nodes in
# the tree as a proxy for model complexity.

G.plotParetoFront()
println("Plotting pareto front. Need to close plot to proceed.")
show()
println("Here are the best solutions from this run:")
for tree in G.globalParetoFront
    println(tree.toString())
end
