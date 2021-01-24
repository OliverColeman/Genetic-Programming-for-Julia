include("GeneticProgramming.jl")
# define input data
X = collect(linspace(0,2*pi,50));
# define output data that we will attempt to reproduce
Y = (X-1).*sin(2*X);
# create a library of nodes that we can use to build
# syntax trees. Deciding which functions and terminals
# to include is problem dependent.
lib = gpLibrary()
for i in -10:10
    lib.addTerminal(i)
end
# add input variable as a terminal.
lib.addTerminal(X)
lib.addFunction(+,2)
lib.addFunction(-,2)
lib.addFunction(*,2)
lib.addFunction(/,2)
lib.addTerminal(pi)
lib.addFunction(sin,1)
lib.addFunction(cos,1)

# define a fitness function. In this case, we will
# use RMSE. 
fitFunct = function(treeOutput)
    sqrt(mean((treeOutput-Y).^2))
end

# Initialize the GP to run in serial. The vars dictionary defines 
# a map that assigns prettier printing to variables that would
# otherwise require many characters to print, such as printing
# a vector or pi.
G = GP(lib,fitFunct,vars=Dict("X"=>X,"pi"=>pi),islandCount=1,popSize=500,mutate=.5);
println("GP Initialized.")
# Run the GP in serial for some desired number of genertations.

G.run(1000)
println("Run Complete.")
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
