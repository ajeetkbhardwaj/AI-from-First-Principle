import Graph

def ucs(graph, startState, endState, isGraph):
    # Initialize open list (f), fWeight (stores f-values), explored nodes (e), and nodePath to store paths
    f = list()
    fWeight = list()
    e = list()
    nodePath = list()
    # Initialize counters 
    expandedNodes = 0
    seenNodes = 0
    maxH = 0
    # Add the start node to the open list (f)
    try:
        f.append(startState)
        seenNodes +=1  # Increment seen nodes counter
        fWeight.append(0) # Initialize f-value with 0 for start node
        nodePath.append([startState]) # Start path is just the start node.
    except:
        print("invalid start state")
    # Main loop to perform UCS search
    while f: # While there are nodes in the open list
        minWeightIndex = fWeight.index(min(fWeight)) # Find the index of the min f-value in fWeight
        tempState = f.pop(minWeightIndex) # Remove that node from the open list
        tempCost = fWeight.pop(minWeightIndex) # Get the cost (g) of the removed node
        tempPath = nodePath.pop(minWeightIndex) # get the path to the removed node
        # If we reached the goal node, return the result
        if tempState == endState: 
            return {"seen": seenNodes, "expanded": expandedNodes, "route": tempPath, "cost": tempCost, "max memory": maxH}
        # Add the expanded node to the expanded list if it's a graph.
        if isGraph:
            e.append(tempState)
        # Exploring the neighbors of the current node.
        expandedNodes += 1
        for node in graph.graph[tempState]:
            # If it's a graph and the node is not in the expanded list (to avoid revisiting), or if it's not a graph:
            if (isGraph and (node not in e)) or (not isGraph):
                # Calculate the cost to reach the neighbor (g) for that neighbor
                cost = graph.getCost(tempState, node)
                tempList = list(tempPath) # copy the current path
                tempList.append(node) # add the neighbor node to the path
                # If the node is already in the open list, check if the new cost is lower
                if node in f:
                    if cost + tempCost < fWeight[f.index(node)]:
                        fWeight[f.index(node)] = cost + tempCost
                        nodePath[f.index(node)] = tempList
                else:
                    f.append(node)
                    seenNodes += 1
                    fWeight.append(cost + tempCost)
                    nodePath.append(tempList)

        maxH = maxH = max(maxH, len(f))

graph1 = [
    [1, 7, 10],          # Node 0: connected to Nodes 1, 7, and 10
    [0, 2, 7],           # Node 1: connected to Nodes 0, 2, and 7
    [1, 3, 5, 8],        # Node 2: connected to Nodes 1, 3, 5, and 8
    [2, 4, 5],           # Node 3: connected to Nodes 2, 4, and 5
    [3, 5],              # Node 4: connected to Nodes 3 and 5
    [2, 3, 4, 6],        # Node 5: connected to Nodes 2, 3, 4, and 6
    [5, 7, 8],           # Node 6: connected to Nodes 5, 7, and 8
    [0, 1, 6, 8],        # Node 7: connected to Nodes 0, 1, 6, and 8
    [2, 6, 7],           # Node 8: connected to Nodes 2, 6, and 7
    [0, 1, 11],          # Node 10: connected to Nodes 0, 1, and 11
    [10, 12, 14],        # Node 11: connected to Nodes 10, 12, and 14
    [11, 15],            # Node 12: connected to Nodes 11 and 15
    [12, 16],            # Node 13: connected to Nodes 12 and 16
    [13, 17],            # Node 14: connected to Nodes 13 and 17
    [14]                 # Node 15: connected to Node 14
]

cost = [
    [4, 8, 15],       # Node 0: costs to Nodes 1, 7, and 10
    [4, 8, 11],       # Node 1: costs to Nodes 0, 2, and 7
    [8, 7, 4, 2],     # Node 2: costs to Nodes 1, 3, 5, and 8
    [7, 9, 14],       # Node 3: costs to Nodes 2, 4, and 5
    [9, 10],          # Node 4: costs to Nodes 3 and 5
    [4, 14, 10, 2],   # Node 5: costs to Nodes 2, 3, 4, and 6
    [2, 1, 6],        # Node 6: costs to Nodes 5, 7, and 8
    [8, 11, 1, 7],    # Node 7: costs to Nodes 0, 1, 6, and 8
    [2, 6, 7],        # Node 8: costs to Nodes 2, 6, and 7
    [4, 8, 12],       # Node 10: costs to Nodes 0, 1, and 11
    [6, 8, 9],        # Node 11: costs to Nodes 10, 12, and 14
    [4, 5],           # Node 12: costs to Nodes 11 and 15
    [10, 7],          # Node 13: costs to Nodes 12 and 16
    [9, 4],           # Node 14: costs to Nodes 13 and 17
    [2]               # Node 15: cost to Node 14
]


myGraph = Graph.Graph(graph1, cost)
print(ucs(myGraph, 0, 8, False))