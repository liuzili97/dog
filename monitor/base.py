class BaseMonitor():

    def init_headers(self):
        raise NotImplementedError

    def update_info(self, name, ssh, is_first_node):
        raise NotImplementedError

    def build_disp(self, all_nodes):
        raise NotImplementedError
