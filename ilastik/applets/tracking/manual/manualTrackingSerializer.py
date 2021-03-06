from ilastik.applets.base.appletSerializer import AppletSerializer,\
    SerialSlot, deleteIfPresent, getOrCreateGroup

class SerialDivisionsSlot(SerialSlot):
    def serialize(self, group):
        if not self.shouldSerialize(group):
            return
        deleteIfPresent(group, self.name)
        group = getOrCreateGroup(group, self.name)
        mainOperator = self.slot.getRealOperator()
        innerops = mainOperator.innerOperators
        for i, op in enumerate(innerops):
            dset = []
            for trackid in op.divisions.keys():
                (children, t_parent) = op.divisions[trackid]
                dset.append([trackid, children[0], children[1], t_parent])
            if len(dset) > 0:
                group.create_dataset(name=str(i), data=dset)
        self.dirty = False

    def deserialize(self, group):
        if not self.name in group:
            return
        mainOperator = self.slot.getRealOperator()
        innerops = mainOperator.innerOperators
        opgroup = group[self.name]
        for inner in opgroup.keys():
            dset = opgroup[inner]            
            op = innerops[int(inner)]
            divisions = {}
            for row in dset:
                divisions[row[0]] = ([row[1],row[2]], row[3])
            op.divisions = divisions
        self.dirty = False
        
class SerialLabelsSlot(SerialSlot):
    def serialize(self, group):
        if not self.shouldSerialize(group):
            return
        deleteIfPresent(group, self.name)
        group = getOrCreateGroup(group, self.name)
        mainOperator = self.slot.getRealOperator()
        innerops = mainOperator.innerOperators
        for i, op in enumerate(innerops):
            gr = getOrCreateGroup(group, str(i))
            for t in op.labels.keys():
                t_gr = getOrCreateGroup(gr, str(t))
                for oid in op.labels[t].keys():
                    l = op.labels[t][oid]
                    dset = list(l)
                    if len(dset) > 0:
                        t_gr.create_dataset(name=str(oid), data=dset)
        self.dirty = False

    def deserialize(self, group):
        if not self.name in group:
            return
        mainOperator = self.slot.getRealOperator()
        innerops = mainOperator.innerOperators
        opgroup = group[self.name]
        for inner in opgroup.keys():
            gr = opgroup[inner]
            op = innerops[int(inner)]
            labels = {}
            for t in gr.keys():
                labels[int(t)] = {}
                t_gr = gr[str(t)]
                for oid in t_gr.keys():
                    labels[int(t)][int(oid)] = set(t_gr[oid])
            op.labels = labels
        self.dirty = False
        
class ManualTrackingSerializer(AppletSerializer):
    
    def __init__(self, operator, projectFileGroupName):
        slots = [ #SerialSlot(operator.TrackImage),
                   SerialDivisionsSlot(operator.Divisions),
                   SerialLabelsSlot(operator.Labels)]
    
        super(ManualTrackingSerializer, self ).__init__(projectFileGroupName, slots=slots)
