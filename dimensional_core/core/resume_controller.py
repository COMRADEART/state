class ResumeController:
    def resolve(self, instance_point):
        if not instance_point:
            return {"global_step": 0, "cycle": 0, "last_eid": 0, "instance": {}, "graphs_state": {}}

        graphs_state = instance_point.get("graphs_state", {})
        if not isinstance(graphs_state, dict):
            graphs_state = {}

        inst = instance_point.get("instance", {})
        if not isinstance(inst, dict):
            inst = {}

        return {
            "global_step": int(instance_point.get("global_step", 0)),
            "cycle": int(instance_point.get("cycle", 0)),
            "last_eid": int(instance_point.get("last_eid", 0)),
            "instance": inst,
            "graphs_state": graphs_state,
        }
