class Graph:
    def __init__(self, graph, weight=None):
        self.edges = list()
        self.nodes = list()
        self.graph = graph
        self.generateEdges(graph, weight)
        self.generateNodes(graph)

    def generateEdges(self, graph, weight):
        if weight:
            for nodeIndex in range(len(graph)):
                for neighbourIndex in range(len(graph[nodeIndex])):
                    self.edges.append((nodeIndex, graph[nodeIndex][neighbourIndex], weight[nodeIndex][neighbourIndex]))
        else:
            for nodeIndex in range(len(graph)):
                for neighbourIndex in range(len(graph[nodeIndex])):
                    self.edges.append((nodeIndex, graph[nodeIndex][neighbourIndex], 0))

    def generateNodes(self, arr):
        for i in range(len(arr)):
            self.nodes.append(i)

    def getCost(self, fromNode, toNode):
        for edge in self.edges:
            if edge[0] == fromNode and edge[1] == toNode:
                return edge[2]


def aStar(graph, startState, endState, h, isGraph):
    # Initialize open list (f), fWeight (stores f-values), explored nodes (e), and nodePath to store paths
    f = list()           # Open list to store nodes to explore
    fWeight = list()     # List to store f-values (g + h) of nodes
    e = list()           # List to store expanded nodes
    nodePath = list()    # List to store the paths to each node

    # Initialize counters
    expandedNodes = 0    # Tracks how many nodes have been expanded
    seenNodes = 0        # Tracks how many nodes have been considered (seen)
    maxH = 0             # Maximum size of the open list during the search

    # Add the start node to the open list (f)
    try:
        f.append(startState)           # Add start node to open list
        seenNodes += 1                 # Increment seen nodes counter
        fWeight.append(h[startState])  # Initialize f-value with heuristic for start node
        nodePath.append([startState])  # Start path is just the start node
    except:
        print("Invalid start state")   # In case there's an issue with the start state

    # Main loop to perform A* search
    while f:
        # Find the node with the minimum f-value (g + h) from open list
        minWeightIndex = fWeight.index(min(fWeight))  # Index of the node with the minimum f-value
        tempState = f.pop(minWeightIndex)              # Remove that node from the open list
        tempCost = fWeight.pop(minWeightIndex) - h[tempState]  # Calculate the cost (g = f - h)
        tempPath = nodePath.pop(minWeightIndex)        # Get the path to the node

        # If we reached the goal node, return the result
        if tempState == endState:
            return {"seen": seenNodes, "expanded": expandedNodes, "route": tempPath, "cost": tempCost, "max memory": maxH}

        # Add the expanded node to the expanded list if it's a graph
        if isGraph:
            e.append(tempState)
        
        # Increment expanded nodes counter
        expandedNodes += 1

        # Explore neighbors of the current node
        for node in graph.graph[tempState]:
            # If it's a graph and the node is not in the expanded list (to avoid revisiting), or if it's not a graph:
            if (isGraph and (node not in e)) or (not isGraph):
                # Calculate the cost to reach the neighbor (g + h for that neighbor)
                cost = graph.getCost(tempState, node) + h[node]
                
                # Create a new list for the current path, appending the neighbor node
                tempList = list(tempPath)  # Copy the current path
                tempList.append(node)      # Add the neighbor node to the path

                # If the node is already in the open list, check if the new cost is lower
                if node in f:
                    # If the new cost is better, update the cost and path
                    if cost + tempCost < fWeight[f.index(node)]:
                        fWeight[f.index(node)] = cost + tempCost
                        nodePath[f.index(node)] = tempList
                else:
                    # If the node is not in the open list, add it
                    f.append(node)
                    seenNodes += 1  # Increment the seen nodes counter
                    fWeight.append(cost + tempCost)  # Add the new f-value (g + h)
                    nodePath.append(tempList)  # Add the new path to the list

        # Track the maximum size of the open list
        maxH = max(maxH, len(f))  # Update maximum size of open list

# Example graph and cost matrices
graph = [[1, 2], [0, 3], [0, 4], [1, 5], [2, 6], [3, 6], [4, 5]]
cost = [[1.5, 2], [1.5, 2], [2, 3], [2, 3], [3, 2], [3, 4], [2, 4]]
h = [8, 4, 4.5, 2, 2, 4, 0]  # Heuristic values for each node

# Create graph object (using a Graph class that should contain methods to retrieve neighbors and costs)
myGraph = Graph.Graph(graph, cost)

# Perform A* search from node 0 to node 6, using the heuristic array `h`, and treating the graph as a tree (isGraph=False)
result = aStar(myGraph, 0, 6, h, False)

# Print the result
print(result)
