from logging import raiseExceptions
import os
import json
from collections import defaultdict
class Info():
    def __init__(self, bench='TrainTicket'):
        
        if bench.lower() == 'trainticket':
            tmp_apiList = ['assurance', 'auth',  'basic', 'cancel', 'config', 'contacts', 'food-map', 'food', 'inside-payment', 
                       'notification', 'order-other', 'order', 'payment', 'preserve', 'price', 'route-plan', 'route', 'seat', 
                       'security', 'station', 'ticketinfo', 'train', 'travel-plan', 'travel', 'travel2', 'user', 'verification-code']
            apiList = ['ts-{}-service'.format(api) for api in tmp_apiList]
            edge_info = {
                    "preserve": ["preserve", "seat", "security", "food", "order", "ticketinfo", "travel", "contacts", "notification", "user", "station"],
                    "seat":["seat", "order", "config", "travel"],   
                    "cancel":["inside-payment", "order-other", "order"],
                    "security": ["security", "order-other", "order"],
                    "food":["travel", "food-map", "station"],
                    "travel": ["travel", "order", "ticketinfo", "train", "route"],
                    "inside-payment":["payment", "order"],
                    "ticketinfo":["ticketinfo", "basic"],
                    "basic":["basic", "route", "price", "train", "station"],
                    "order-other":["station"],
                    "order":["order", "station", "assurance"],
                    "auth":["auth", "verification-code"]
                    }
            self.edge_info = {'ts-{}-service'.format(k):['ts-{}-service'.format(vi) 
                                                        for vi in v] for k,v in edge_info.items()} 

        elif bench.lower() == 'socialnetwork':
            apiList = ['social-graph-service', 'compose-post-service', 'post-storage-service', 'user-timeline-service', 'url-shorten-service', 'user-service',
                       'media-service', 'text-service', 'unique-id-service', 'user-mention-service', 'home-timeline-service', "nginx-web-server"]
            self.edge_info = {
                "compose-post-service": ["compose-post-service", "home-timeline-service", "media-service", "post-storage-service", 
                                        "text-service", "unique-id-service", "user-service", "user-timeline-service"],
                "home-timeline-service":["home-timeline-service", "post-storage-service", "social-graph-service"],
                "post-storage-service": ["post-storage-service"],
                "social-graph-service": ["social-graph-service", "user-service"],
                "text-service": ["text-service", "url-shorten-service", "user-mention-service"],
                "user-service": ["user-service"],
                "user-timeline-service": ["user-timeline-service"],
                "nginx-web-server": ["compose-post-service", "home-timeline-service", "nginx-web-server", "social-graph-service", "user-service"]
            }
        
        elif bench.lower() == 'gaia':
            # Gaia数据集：10个服务 (基于预处理结果)
            apiList = ['dbservice1', 'dbservice2', 'logservice1', 'logservice2', 
                        'mobservice1', 'mobservice2', 'redisservice1', 'redisservice2', 
                        'webservice1', 'webservice2']
            
            # 从TVDiag预处理结果中动态加载真实的服务依赖关系
            self.edge_info = load_gaia_edges_from_tvdiag()
        
        else:
            raiseExceptions("Not Implemented yet {}".format(bench))
        
        # 配置指标名称 (根据数据集不同设置不同维度的指标)
        if bench.lower() == 'gaia':
            # Gaia数据集：12维指标 (基于TVDiag预处理的metric文件)
            self.metric_names = [
                'cpu_total_norm_pct', 'diskio_read_bytes', 'diskio_read_service_time',
                'diskio_write_bytes', 'diskio_write_service_time', 'memory_fail_count',
                'memory_usage_pct', 'network_in_bytes', 'network_in_errors',
                'network_out_bytes', 'network_out_dropped', 'network_out_errors'
            ]
        else:
            # SN/TT数据集：7维指标
            self.metric_names = ['cpu_usage_system', 'cpu_usage_total', 'cpu_usage_user', 
                               'memory_usage', 'memory_working_set', 'rx_bytes', 'tx_bytes']
       
        self.service_names = apiList
        self.service2nid = {s:idx for idx, s in enumerate(self.service_names)}
        self.node_num = len(self.service_names)
        
        self.metadata = {
            "node_num": self.node_num,
            "metric_num": len(self.metric_names),
        }
        self.__get_edges()
    
    def __get_edges(self):
        src, des = [], []
        for s, v in self.edge_info.items():
            sid = self.service2nid[s]
            for t in v:
                src.append(sid)
                des.append(self.service2nid[t])
        self.edges = [src, des]
    
    def add_info(self, key, value):
        self.metadata[key] = value

def read_json(filepath):
    if os.path.exists(filepath):
        assert filepath.endswith('.json')
        with open(filepath, 'r') as f:
            return json.loads(f.read())
    else: 
        raiseExceptions("File path "+filepath+" not exists!")
        return

def json_pretty_dump(obj, filename):
    with open(filename, "w") as fw:
        json.dump(obj,fw, sort_keys=True, indent=4, separators=(",", ": "), ensure_ascii=False)


def load_gaia_edges_from_tvdiag():
    """
    从TVDiag预处理的图结果中加载Gaia的真实服务依赖关系
    
    Returns:
        dict: 服务依赖关系映射 {service: [dependent_services]}
    """
    # 获取脚本目录和项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    eadro_root = os.path.dirname(os.path.dirname(script_dir))
    project_root = os.path.dirname(os.path.dirname(eadro_root))
    
    # TVDiag预处理的graph结果路径
    graph_dir = os.path.join(project_root, 'data', 'processed_data', 'gaia', 'graph')
    edges_file = os.path.join(graph_dir, 'edges_static_no_influence.json')
    nodes_file = os.path.join(graph_dir, 'nodes_static_no_influence.json')
    
    # 加载nodes和edges（格式为 {sample_id: data}）
    with open(nodes_file, 'r') as f:
        nodes_data = json.load(f)
    with open(edges_file, 'r') as f:
        edges_data = json.load(f)
    
    # 从第一个样本中提取数据（static模式下所有样本的图结构都相同）
    sample_id = list(nodes_data.keys())[0]
    nodes = nodes_data[sample_id]  # 服务名列表：['dbservice1', 'dbservice2', ...]
    edges = edges_data[sample_id]  # 索引对列表：[[8, 2], [3, 2], ...]
    
    # 构建服务依赖关系
    edge_info = defaultdict(set)
    
    # 将数字索引转换为服务名
    for source_idx, target_idx in edges:
        source_service = nodes[source_idx]  # 索引转服务名
        target_service = nodes[target_idx]  # 索引转服务名
        
        edge_info[source_service].add(target_service)
    
    # 确保每个服务至少包含自己
    gaia_services = ['dbservice1', 'dbservice2', 'logservice1', 'logservice2', 
                    'mobservice1', 'mobservice2', 'redisservice1', 'redisservice2', 
                    'webservice1', 'webservice2']
    
    for service in gaia_services:
        edge_info[service].add(service)  # 每个服务包含自己
    
    # 转换为list格式
    return {k: list(v) for k, v in edge_info.items()}


if __name__ == "__main__":
    info = Info('gaia')
