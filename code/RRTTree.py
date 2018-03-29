import operator

class RRTTree(object):
    
    def __init__(self, planning_env, start_config, start_cost):
        
        self.planning_env = planning_env
        self.vertices = []
        self.vertices.append(start_config)
        self.costs = []
        self.costs.append(start_cost)
        self.edges = dict()

        self.max_cost = 0

    def GetRootId(self):
        return 0

    def GetNearestVertex(self, config):
        
        dists = []
        for v in self.vertices:
            dists.append(self.planning_env.ComputeDistance(config, v))

        vid, vdist = min(enumerate(dists), key=operator.itemgetter(1))

        return vid, self.vertices[vid], self.costs[vid]
            
    def get_max_cost(self):
        if len(self.costs) > 0:
            return max(self.costs)
        else:
            return 0            
        # return self.max_cost

    def AddVertex(self, config, cost):
        vid = len(self.vertices)
        self.vertices.append(config)
        self.costs.append(cost)
        self.max_cost = max(self.costs)
        return vid

    def AddEdge(self, sid, eid):
        self.edges[eid] = sid

    def getParent(self, eid):
        try:
            return self.edges[eid], self.vertices[self.edges[eid]]
        except KeyError, e:
            return None, None
