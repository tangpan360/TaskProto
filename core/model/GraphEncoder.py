
from torch import nn
from core.model.backbone.sage import SAGEEncoder
# from core.model.backbone.tag import TAGEncoder
# from core.model.backbone.SGC import SGCEncoder


class GraphEncoder(nn.Module):
    def __init__(self, 
                feature_embedding_dim: int,
                graph_hidden_dim: int, 
                graph_out_dim: int,
                num_layers=2,
                aggregator='mean',
                feat_drop=0.3):
        super(GraphEncoder, self).__init__()

        self.graph_encoder = SAGEEncoder(
            in_dim=feature_embedding_dim,
            out_dim=graph_out_dim,
            hidden_dim=graph_hidden_dim,
            num_layers=num_layers,
            aggregator_type=aggregator,
            feat_drop=feat_drop
        )

        # self.graph_encoder = TAGEncoder(
        #     in_dim=feature_embedding_dim,
        #     out_dim=graph_out_dim,
        #     hidden_dim=graph_hidden_dim,
        #     num_layers=num_layers
        # )
        # self.graph_encoder = SGCEncoder(
        #     in_dim=feature_embedding_dim,
        #     out_dim=graph_out_dim,
        #     hidden_dim=graph_hidden_dim,
        #     num_layers=num_layers
        # )


    def forward(self, g, x):
        f, e = self.graph_encoder(g, x)
        return f, e
        