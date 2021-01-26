#module GeneticProgramming
using Distributed
using PyPlot
using Interact


mutable struct gpLibrary
    terminals::Array{Any,1}
    functions::Array{Function,1}
    numInputs::Array{Int64,1}
    addTerminal::Function
    addFunction::Function
    randomTerminal::Function
    randomFunction::Function
    randomVariable::Function
    
    function gpLibrary()
        this = new()
        this.terminals = []
        this.functions = []
        this.numInputs = []
        
        this.addTerminal = function(s) #::Symbol)
            push!(this.terminals,s)
        end
        
        this.addFunction = function(f::Function,num::Int)
            push!(this.functions,f)
            push!(this.numInputs,num)
        end
            
        this.randomTerminal = function()
            return this.terminals[rand(1:end)]
        end
        
        this.randomFunction = function()
            r = rand(1:length(this.functions))
            return this.functions[r],this.numInputs[r]
        end
        
        this.randomVariable = function()
            numTerminals = length(this.terminals)
            numVars = numTerminals+length(this.functions)
            r = rand(1:numVars)
            #println((r,numTerminals))
            if r <= numTerminals
                return this.terminals[r],0
            end
            return this.functions[r-numTerminals],this.numInputs[r-numTerminals]
        end
        
        return this
    end    
end


mutable struct Tree
    root
    depth::Int
    age::Int
    numNodes::Int
    fitness::Float64
    nodes::Array{Any,1}
    eval::Function
    toString::Function
    #toSimpleString::Function
    copy::Function
    listNodes::Function
    equals::Function
    dominates::Function
    variables::Dict
    calcDepth::Function
    
    function Tree(maxDepth::Int,library::gpLibrary,vars::Dict)
        this = new()
        this.fitness = -1
        this.age = 0
        this.depth = 0
        this.numNodes = 0
        this.nodes = Any[]
        this.variables = vars
        
        this.calcDepth = function()
            this.numNodes = 0
            driver = function(node,currentDepth)
                this.numNodes += 1
                if typeof(node) != Expr
                    return currentDepth
                end
                depths = Int[]
                for i in 2:length(node.args)
                    push!(depths,driver(node.args[i],currentDepth+1))
                end
                return maximum(depths)
            end
            this.depth = driver(this.root,0)
            return this.depth
        end
        
        this.listNodes = function(force)
            if force || length(this.nodes) == 0
                driver = function(node,depth)
                    out = Tuple[]
                    push!(out,(node,depth))
                    if typeof(node) == Expr
                        for i=2:length(node.args)
                            append!(out,driver(node.args[i],depth+1))
                        end
                    end
                    return out
                end
                
                nodes = driver(this.root,0)
                this.depth = 0
                this.nodes = Any[]
                for N in nodes
                    node = N[1]
                    depth = N[2]
                    if depth > this.depth
                        this.depth = depth
                    end
                    #push!(this.nodes,node)
                end
                append!(this.nodes,[n[1] for n in nodes])
                this.numNodes = length(this.nodes)
            end
            return this.nodes 
        end
        
        
        if maxDepth == 0
            this.root = library.randomTerminal()
            this.depth = 0
            this.numNodes = 1
            push!(this.nodes,this.root)
        elseif maxDepth > 0
            # use a dummy initial expr, and update it
            this.root = :(sin(0))
            f,numArgs = library.randomFunction()
            this.root.args[1] = f
            this.root.args[2] = library.randomTerminal()
            for i in 2:numArgs
                push!(this.root.args,library.randomTerminal())
            end
            currentNodes = Tuple[]
            push!(currentNodes,(this.root,1))
            while length(currentNodes) > 0
                node, depthOfChildren = popfirst!(currentNodes)
                if depthOfChildren > this.depth
                    this.depth = depthOfChildren
                end
                if depthOfChildren < maxDepth
                    for i=2:length(node.args)
                        V,numArgs = library.randomVariable()
                        if numArgs > 0
                            # use dummy expr again and update
                            newExpr = :(sin(0))
                            newExpr.args[1] = V
                            newExpr.args[2] = library.randomTerminal()
                            for j=2:numArgs
                                push!(newExpr.args,library.randomTerminal())
                            end
                            node.args[i] = newExpr
                            push!(currentNodes,(newExpr,depthOfChildren+1))
                        end
                    end
                end
            end
            this.listNodes(true)
        end
        
        this.eval = function()
            return eval(@.(this.root))
        end
        
        this.toString = function()

            driver = function(e)
                if typeof(e) == Expr
                    out = string(e.args[1])*"("*driver(e.args[2])
                    for i=3:length(e.args)
                        out *= ","*driver(e.args[i])
                    end
                    out *= ")"
                else
                    out = string(e)
                    for kv in this.variables
                        if kv[2] == e
                            out = kv[1]
                        end
                    end
                end
                return out
            end
            
            output = driver(this.root)
            return output
        end
        
        
        this.copy = function()
            newTree = Tree(-1,gpLibrary(),this.variables)
            newTree.root = copy(this.root)
            newTree.depth = this.depth
            newTree.fitness = this.fitness
            newTree.age = this.age
            newTree.listNodes(true)
            return newTree
        end
        
        this.equals = function(tree::Tree)
            if this.numNodes != tree.numNodes || this.depth != tree.depth
                return false
            end
            for i in 1:length(this.nodes)
                if this.nodes[i] != tree.nodes[i]
                    return false
                end
            end
            return true
        end
        
        this.dominates = function(tree::Tree)
            # returns a boolean indicating if 'this' 
            # dominates tree.
            if this.fitness <= tree.fitness && this.numNodes <= tree.numNodes && this.age <= tree.age
                if this.fitness != tree.fitness || this.numNodes != tree.numNodes
                    return true
                end
                # not an improvement if equivalent.
                return false
            end
            return false
        end
        
        return this
    end
end


mutable struct Population
    forest::Array{Tree,1}
    popSize::Int
    babyPopSize::Int
    maxDepth::Int
    numGen::Int
    paretoFront::Array{Tree,1}
    library::gpLibrary
    fitnessFunction::Function
    mutationRate::Float64
    mutateTree::Function
    crossTrees::Function
    iterate::Function
    calcPareto::Function
    #plotParetoFront::Function
    simplifiedParetoFront::Function
    #exploreParetoFront::Function
    variables::Dict
    
    function Population(popSize::Int,lib::gpLibrary,fit::Function;mutate=.5,vars=Dict())
        this = new()
        this.numGen = 0
        this.babyPopSize = round(Int,popSize/10)
        this.maxDepth = 10
        this.mutationRate = mutate
        this.fitnessFunction = fit
        this.popSize = popSize
        this.paretoFront = Tree[]
        this.library = lib
        this.variables = vars
        
        this.calcPareto = function(newTrees)
            # change to only assess new trees
            #this.paretoFront = Tree[]
            for currentTree in newTrees
                # Assess tree for fitness.
                try
                    if currentTree.depth > this.maxDepth
                        currentTree.fitness = Inf
                    elseif currentTree.fitness == -1
                        out = currentTree.eval()
                        currentTree.fitness = this.fitnessFunction(out)
                    end
                catch e
                    #println("tree eval error ", e)
                    # Flag tree for removal if errors.
                    currentTree.fitness = Inf
                end
            
                # Not a candidate for pareto front if flagged for removal
                if !isnan(currentTree.fitness) && currentTree.fitness < Inf
                    if length(this.paretoFront) == 0
                        # First tree gets into pareto front trivially.
                        push!(this.paretoFront,currentTree)
                    else
                        # For each tree already in the pareto front,
                        # make sure the pareto-tree has lower
                        # fitness, depth, and age than current tree.
                        # If current tree is undominated by a tree 
                        # already in the pareto front, then add the 
                        # current tree to the pareto front.
                        # If current tree actually dominates a pareto-tree,
                        # then remove the pareto-tree from the pareto front.
                        dominated = false
                        index = 1
                        while index <= length(this.paretoFront)
                            paretoTree = this.paretoFront[index]
                            #if currentTree.fitness == paretoTree.fitness &&
                            #    currentTree.depth == paretoTree.depth && 
                            #    currentTree.age == paretoTree.age
                            #    break
                            #else
                            if currentTree.dominates(paretoTree)
                                # remove pareto-tree
                                splice!(this.paretoFront,index)
                            elseif paretoTree.dominates(currentTree)
                                    # currentTree is dominated, and 
                                    # is not a candidate for the 
                                    # pareto front.
                                    dominated = true
                                    break
                            elseif currentTree.fitness == paretoTree.fitness &&
                                    currentTree.numNodes == paretoTree.numNodes
                                # Ties disqualify the currentTree as well.
                                dominated = true
                                break
                            else 
                                index += 1
                            end
                        end
                        if !dominated
                            push!(this.paretoFront,currentTree)
                        end
                    end
                end
            end
        end

        this.forest = Tree[]
        for i=1:this.popSize
            push!(this.forest,Tree(rand(0:3),this.library,this.variables))
        end
        this.calcPareto(this.forest)

        
        this.mutateTree = function(parentTree::Tree)
            # randomly mutate the gpVariable stored
            # at a randomly selected node in the given tree.
            # Correct the number of children for the node
            # to produce the correct number of inputs for the gpVariable.
            tree = parentTree.copy()
            index = rand(1:length(tree.nodes))
            node = tree.nodes[index]
            if typeof(node) != Expr
                terminal = this.library.randomTerminal()
                index -= 1
                while index >= 1
                    temp = tree.nodes[index]
                    if typeof(temp) == Expr && node in temp.args
                        for j in 2:length(temp.args)
                            if temp.args[j] == node
                                temp.args[j] = terminal
                                break
                            end
                        end
                        break
                    end
                    index -= 1
                end
                node = terminal
            else
                F,numInputs = this.library.randomFunction()
                currentNum = length(node.args)-1
                if numInputs < currentNum
                    node.args = node.args[1:numInputs+1]
                else
                    for i in 1:(numInputs-currentNum)
                        push!(node.args,this.library.randomTerminal())
                    end
                end
                node.args[1] = F
            end
            tree.listNodes(true)
            tree.fitness = -1
            return tree
        end
        
        this.crossTrees = function(parent1::Tree,parent2::Tree)
            # For each parent, randomly select node as point at which to
            # perform cross over. Return the child.
            child = parent1.copy()
            nodes1 = child.listNodes(true)
            nodes2 = parent2.listNodes(false)
            
            newNode = nodes2[rand(1:end)] 
            index = rand(1:length(nodes1))
            oldNode = nodes1[index]
            index -= 1
            while index >= 1
                temp = nodes1[index]
                if typeof(temp) == Expr && oldNode in temp.args
                    for j in 2:length(temp.args)
                        if temp.args[j] == oldNode
                            temp.args[j] = newNode
                            break
                        end
                    end
                    break
                end
                index -= 1
            end
            if index == 0
                # There does not exist a parent node
                # to the root of parent-tree 1.
                return parent2.copy()
            end
            child.age = max(parent1.age,parent2.age)
            child.calcDepth();
            child.fitness = -1
            return child
        end
        
        this.iterate = function()
            #iterate over generation
            # 1. for each spot in population,
            #    create new tree by mutation or cross production.
            # 2. update pareto front. Also updates complexity and fitness
            #############
            # possible spot to embarassingly parallelize...
            # @parallel
            
            for tree in this.forest
                tree.age += 1
            end
            newGen = Tree[]
            potentialParents = Tree[]
            # adds randomness/exploration to the population.
            # Maybe allow tuning this as function of iteration.
            for i in 1:this.babyPopSize
                tree = Tree(rand(1:3),this.library,this.variables)
                push!(newGen,tree)
            end
            append!(potentialParents,this.paretoFront)
            append!(potentialParents,newGen)
            #append!(newGen,this.paretoFront)
            count = length(potentialParents) #newGen)
            this.popSize = max(count,this.popSize)
            R = rand(1,this.popSize-count) #this.babyPopSize-length(this.paretoFront))
            for r in R
                if r < this.mutationRate
                    push!(newGen,this.mutateTree(potentialParents[rand(1:count)]))
                else
                    # changed from newGen
                    parent1 = potentialParents[rand(1:count)]
                    parent2 = potentialParents[rand(1:count)]
                    push!(newGen,this.crossTrees(parent1,parent2))
                end
            end
            this.forest = newGen
            this.calcPareto(newGen)
            this.numGen += 1
        end
        
        this.simplifiedParetoFront = function()
            if length(this.paretoFront) == 0
                return []
            end
            nodeCounts = [t.numNodes for t in this.paretoFront]
            m = minimum(nodeCounts)
            M = maximum(nodeCounts)
            lastFit = Inf
            simpleParetoFront = []
            for i in m:M
                temp = []
                for t in this.paretoFront
                    if t.numNodes == i
                        push!(temp,t)
                    end
                end
                if length(temp) > 0
                    minFitTree = temp[1]
                    for t in temp[2:end]
                        if t.fitness < minFitTree.fitness
                            minFitTree = t
                        end
                    end
                    if minFitTree.fitness <= lastFit
                        push!(simpleParetoFront,minFitTree)
                        lastFit = minFitTree.fitness
                    end
                end
            end
            return simpleParetoFront
        end
        
#         this.plotParetoFront = function()
#             simpleP = this.simplifiedParetoFront()
#             X = [t.numNodes for t in simpleP]
#             Y = [t.fitness for t in simpleP]
#             scatter(X,Y)
#             xlabel("Number of Nodes")#,fontsize=15)
#             ylabel("Fitness")#,fontsize=15)
#             t = string("Generation: ",this.numGen)
#             title(t)
#         end
#        
#         this.exploreParetoFront = function(input,targetOutput)
#             # only for symbolic regression
#             simpleP = P.simplifiedParetoFront()
#             f = figure()
#             @manipulate for index in 1:length(simpleP)
#                 withfig(f) do
#                     Z = simpleP[index].eval()
#                     plot(input,targetOutput,"k-",label="Target")
#                     if length(Z) == 1
#                         Z = Z*ones(length(input))
#                     end
#                     plot(input,Z,"r-",label="Solution")
#                     title(@sprintf "%s,\n Fitness=%0.3f, Depth=%d, #Nodes=%d" simpleP[index].toString() simpleP[index].fitness simpleP[index].depth simpleP[index].numNodes)
#                     legend()
#                 end
#             end
#         end

        return this
    end
end


mutable struct GP
    islandCount::Int
    islands::Array{Any,1}
    popSize::Int
    fitnessFunct::Function
    library::gpLibrary
    vars::Dict
    mutationRate::Float64
    migrationRate::Float64
    numMigrate::Int
    run::Function
    globalParetoFront::Array{Tree,1}
    getGlobalParetoFront::Function
    plotParetoFront::Function
    exploreParetoFront::Function
    migrate::Function
    numGen::Int
    tolerance::Float64
    numNewTree::Int

    function GP(lib::gpLibrary,fit::Function;mutate=.5,vars=Dict(),popSize=100,islandCount=1,migrate=100,numMigrate=2,tol=.1,numNewTree=-1)
        this = new()
        this.popSize = popSize
        this.fitnessFunct = fit
        this.library = lib
        this.vars = vars
        this.mutationRate = mutate
        this.migrationRate = migrate
        this.numMigrate = numMigrate
        this.islandCount = islandCount
        this.globalParetoFront = Tree[]
        this.numGen = 0
        this.islands = Any[]
        this.tolerance = tol
        if numNewTree == -1
            this.numNewTree = round(Int,popSize/10)
        else
            this.numNewTree = numNewTree
        end
       
        if this.islandCount > 1
            n = nprocs()
            this.islandCount = minimum([n-1,this.islandCount])
        end
        if this.islandCount > 1
            println("Running in parallel")
            # initialize library and fit funct
            # on workers. Note: Julia seems to 
            # be unable to serialize user-defined
            # datatypes, so we need to accomplish
            # this by communicating only Julia 
            # natives. This incurs a strong limitation
            # on what objects can comprise the gpLibrary.

            r = RemoteRef()
            # Make sure types are defined everywhere.

            #@everywhere begin
            #    if myid() != 1
            #        include("GeneticProgramming.jl")
            #    end
            #end
            
            # build the library everywhere.
            put!(r,this.library)
            @sync for p in procs()
                @spawnat p eval(Main,Expr(:(=),:library,fetch(r)))
            end
            take!(r)
            

            put!(r,this.popSize)
            #println(fetch(r))
            #println(procs())
            @sync for p in procs()
                @spawnat p eval(Main,Expr(:(=),:pSize,fetch(r)))
            end
            take!(r);

            put!(r,this.fitnessFunct)
            #println(fetch(r))
            #println(procs())
            @sync for p in procs()
                @spawnat p eval(Main,Expr(:(=),:fit,fetch(r)))
            end
            take!(r)

            put!(r,this.vars)
            #println(fetch(r))
            #println(procs())
            @sync for p in procs()
                @spawnat p eval(Main,Expr(:(=),:Vars,fetch(r)))
            end 
            take!(r)

            put!(r,this.mutationRate)
            #println(fetch(r))
            #println(procs())
            @sync for p in procs()
                @spawnat p eval(Main,Expr(:(=),:mut,fetch(r)))
            end 
            take!(r)

            put!(r,this.tolerance)
            #println(fetch(r))
            #println(procs())
            @sync for p in procs()
                @spawnat p eval(Main,Expr(:(=),:tolerance,fetch(r)))
            end 
            take!(r)
            
            #println("What's in local memory?")
            #for p in procs()
            #    @sync @spawnat p whos()
            #    println("---------------------------------------")
            #end

            put!(r,this.numMigrate)
            @sync for p in procs()
                @spawnat p eval(Main,Expr(:(=),:numMigrate,fetch(r)))
            end
            take!(r)

            put!(r,this.numNewTree)
            @sync for p in procs()
                @spawnat p eval(Main,Expr(:(=),:numNewTree,fetch(r)))
            end
            take!(r)

            Rs = [RemoteRef() for p in procs()]
            put!(r,Rs)
            @sync for p in procs()
                @spawnat p eval(Main,Expr(:(=),:Rs,fetch(r)))
            end
            take!(r)

            @everywhere begin
                P = Population(pSize,library,fit,mutate=mut,vars=Vars)
                P.babyPopSize = numNewTree
            end
            
            #@sync for p in procs()
            #    @spawnat p eval(Main, Expr(:(=),:success,false))
            #end

            println("Populations intialized.")
        else
            # if serial
            println("Running in Serial.")
            push!(this.islands,Population(this.popSize,this.library,this.fitnessFunct,mutate=this.mutationRate,vars=this.vars))
        end

        this.getGlobalParetoFront = function()
            if this.islandCount == 1
                # if serial
                this.globalParetoFront = this.islands[1].simplifiedParetoFront()
            else
                # if in parallel
                this.globalParetoFront = Tree[]
                trees = Tree[]
                PF = Tree[]
                @sync for p in procs()
                    @spawnat p put!(Rs[myid()],P.paretoFront);
                end
                for r in Rs
                    append!(trees,take!(r));
                end
                for currentTree in trees
                    if length(this.globalParetoFront) == 0
                        # First tree gets into pareto front trivially.
                        push!(PF,currentTree)
                    else
                        # For each tree already in the pareto front,
                        # make sure the pareto-tree has lower
                        # fitness, depth, and age than current tree.
                        # If current tree is undominated by a tree 
                        # already in the pareto front, then add the 
                        # current tree to the pareto front.
                        # If current tree actually dominates a pareto-tree,
                        # then remove the pareto-tree from the pareto front.
                        dominated = false
                        index = 1
                        while index <= length(PF)
                            paretoTree = PF[index]
                            #if currentTree.fitness == paretoTree.fitness &&
                            #    currentTree.depth == paretoTree.depth && 
                            #    currentTree.age == paretoTree.age
                            #    break
                            #else
                            if currentTree.dominates(paretoTree)
                                # remove pareto-tree
                                splice!(PF,index)
                            elseif paretoTree.dominates(currentTree)
                                    # currentTree is dominated, and 
                                    # is not a candidate for the 
                                    # pareto front.
                                    dominated = true
                                    break
                            elseif currentTree.fitness == paretoTree.fitness &&
                                    currentTree.numNodes == paretoTree.numNodes
                                # Ties disqualify the currentTree as well.
                                dominated = true
                                break
                            else 
                                index += 1
                            end
                        end
                        if !dominated
                            push!(PF,currentTree)
                        end
                    end
                end                
                if length(PF) == 0
                    this.globalParetoFront = Tree[]
                    return Tree[]
                end
                nodeCounts = [t.numNodes for t in PF]
                m = minimum(nodeCounts)
                M = maximum(nodeCounts)
                lastFit = Inf
                simpleParetoFront = []
                for i in m:M
                    temp = []
                    for t in PF
                        if t.numNodes == i
                            push!(temp,t)
                        end
                    end
                    if length(temp) > 0
                        minFitTree = temp[1]
                        for t in temp[2:end]
                            if t.fitness < minFitTree.fitness
                                minFitTree = t
                            end
                        end
                        if minFitTree.fitness <= lastFit
                            push!(simpleParetoFront,minFitTree)
                            lastFit = minFitTree.fitness
                        end
                    end
                end
                this.globalParetoFront = simpleParetoFront
            end
            return this.globalParetoFront
        end
        

        this.migrate = function()
            #println("Migrating.....")
            if this.islandCount == 1
                return
            end
            @everywhere begin
                temp = Tree[]
                numPareto = length(P.paretoFront)
                if numPareto > 0
                    for i in 1:numMigrate
                        push!(temp,P.paretoFront[rand(1:numPareto)])
                    end
                end
                put!(Rs[((myid()+1) % nprocs())+1],temp)
            end

            @everywhere begin
                temp = take!(Rs[myid()])
                for tree in temp
                        push!(P.paretoFront,tree)
                end
            end
        end

        
        this.run = function(numberOfGenerations)
            if this.islandCount == 1
                # if serial
                P = this.islands[1]
                for i in 1:numberOfGenerations
                    P.iterate()
                end
                this.numGen += numberOfGenerations
            else
                # if parallel
                goodEnough = false
                r = RemoteRef()
                num = round(Int,numberOfGenerations/this.migrationRate)

                put!(r,this.migrationRate)
                @sync for p in procs()
                    @spawnat p eval(Main,Expr(:(=),:numToIterate,fetch(r)))
                end
                take!(r)                

                for I in 1:num
                    @everywhere begin
                        for i in 1:numToIterate
                            P.iterate()
                        end
                        success = false
                        for tree in P.paretoFront
                            if tree.fitness < tolerance
                                success = true
                                break
                            end
                        end
                        put!(Rs[myid()],success)
                    end
                    for r in Rs
                        if take!(r)
                            goodEnough = true
                        end
                    end
                    this.numGen += this.migrationRate
                    println(this.numGen)
                    if goodEnough
                        println("Perfect Solution Found!")
                        break
                    else
                        this.migrate()
                    end
                end

                num = numberOfGenerations-num*this.migrationRate
                if num > 0 && !goodEnough
                    put!(r,num)
                    @sync for p in procs()
                        @spawnat p eval(Main,Expr(:(=),:numToIterate,fetch(r)))
                    end
                    take!(r)                                

                    @everywhere begin
                        for i in 1:numToIterate
                            P.iterate()
                        end
                    end
                    this.numGen += num
                end
            end
            return this.getGlobalParetoFront()
        end

        this.plotParetoFront = function()
            counts = [t.numNodes for t in this.globalParetoFront]
            fits = [t.fitness for t in this.globalParetoFront]
            scatter(counts,fits)
            xlabel("Number of Nodes")#,fontsize=15)
            ylabel("Fitness")#,fontsize=15)
            t = string("Generation: ",this.numGen)
            title(t)
        end
        
        this.exploreParetoFront = function(input,targetOutput)
            # only for symbolic regression
            f = figure()
            @manipulate for index in 1:length(this.globalParetoFront)
                withfig(f) do
                    tree = this.globalParetoFront[index]
                    Z = tree.eval()
                    plot(input,targetOutput,"k-",label="Target")
                    if length(Z) == 1
                        Z = Z*ones(length(input))
                    end
                    plot(input,Z,"r-",label="Solution")
                    title("$(tree.toString()),\n Fitness=$(tree.fitness), Depth=$(tree.depth), #Nodes=$(tree.numNodes)")
                    legend()
                end
            end
        end

        return this
    end
end
#end 

#end