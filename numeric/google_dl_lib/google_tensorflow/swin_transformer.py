class MLP(keras.Layer):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        C = input_shape[-1]
        self.net = keras.Sequential(
            [
                layers.Dense(C * 4),
                layers.Activation("gelu"),
                layers.Dropout(0.1),
                layers.Dense(C),
            ]
        )
        return super().build(input_shape)

    def call(self, x):
        return self.net(x)

class MSA(keras.Layer):
    def __init__(self, heads=8, **kwargs):
        super().__init__(**kwargs)
        self.heads = heads
        self.head_dim = None
        
    def build(self, input_shape):
        C = input_shape[-1]
        self.linear = layers.Dense(C)
        
        self.head_dim = C // self.heads
        self.qhead = layers.Dense(C, use_bias=False)
        self.khead = layers.Dense(C, use_bias=False)
        self.vhead = layers.Dense(C, use_bias=False)
        return super().build(input_shape)

    def call(self, x, mask=None):
        x = ops.transpose(x, (1, 0, 2, 3))
        x = ops.vectorized_map(self.vectorized_msa_window, (x, mask))
        x = ops.transpose(x, (1, 0, 2, 3))
        return x
    
    def vectorized_msa_window(self, arg):
        x, mask = arg
        # B, S, C
        bz, sz, C = ops.shape(x)
        C = x.shape[-1]
        q, k, v = x, x, x #self.qf(x), self.kf(x), self.vf(x)
        
        qh = self.qhead(q)
        kh = self.khead(k)
        
        qh = ops.reshape(qh, (bz, sz, self.heads, self.head_dim))
        qh = ops.transpose(qh, (0, 2, 1, 3))
        kh = ops.reshape(kh, (bz, sz, self.heads, self.head_dim))
        kh = ops.transpose(kh, (0, 2, 1, 3))
        
        inn_prod = ops.matmul(qh, ops.transpose(kh, (0, 1, 3, 2)))
        scaled_inn_prod = inn_prod / ops.sqrt(self.head_dim)
        
        attn_weights = ops.softmax(scaled_inn_prod, -1)
        mask = ops.broadcast_to(mask, ops.shape(attn_weights)) 
        masked_attn_weights = ops.multiply(attn_weights, mask)
        
        vh = self.vhead(x)
        vh = ops.reshape(vh, (bz, sz, self.heads, self.head_dim))
        vh = ops.transpose(vh, (0, 2, 1, 3))  
        
        x = ops.matmul(masked_attn_weights, vh)
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (bz, sz, C))
        x = self.linear(x)
        x = tf.nn.dropout(x, 0.1)
        return x
        
    
    def forward_vectorized(self, x, mask):
        pass

class SWMSA(MSA):
    def __init__(self, window_size, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        
    def build(self, input_shape):
        bz, h, w, C = input_shape
        # mask = np.zeros((h, w)).astype(int)
        # self.partition_layer = PartitionLayer(self.window_size)
        # self.merge_layer = PartitionMerging(h // self.window_size)
        # self.reshape = layers.Reshape(( 
        #                                (h // self.window_size) * (w // self.window_size),
        #                             self.window_size * self.window_size, 
        #                             C))
        # super().build(input_shape)

    def call(self, x, mask=None):
        bz, h, w, C = ops.shape(x)
        left, right = np.floor(self.window_size / 2), np.ceil(self.window_size / 2)
        pad = None
        if right - left != 0:
            pad = abs(right - left)
        shift_length = min(left, right)
        
        x = ops.roll(x, (-shift_length, -shift_length), (1, 2))
        
        raise Exception
        
        x = self.partition_layer(x)

        x = self.reshape(x)
        seq_length = self.window_size ** 2

        mask = ops.ones((seq_length, seq_length))
        mask = ops.expand_dims(mask, 0)
        x = super().call(x, mask)
        x = ops.reshape(x, (bz, x.shape[1], self.window_size, self.window_size, C))
        x = ops.reshape(x, (bz, int(np.sqrt(x.shape[1])), int(np.sqrt(x.shape[1])), 
                            self.window_size, self.window_size, C
                            ))
        
        x = self.merge_layer(x)
        x = ops.squeeze(x, (1, 2))
        return x

class WMSA(MSA):
    def __init__(self, window_size, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        
    def build(self, input_shape):
        bz, h, w, C = input_shape
        self.partition_layer = PartitionLayer(self.window_size)
        self.merge_layer = PartitionMerging(h // self.window_size)
        self.reshape = layers.Reshape(( 
                                       (h // self.window_size) * (w // self.window_size),
                                    self.window_size * self.window_size, 
                                    C))
        super().build(input_shape)

    def call(self, x, mask=None):
        bz, h, w, C = ops.shape(x)
        x = self.partition_layer(x)

        x = self.reshape(x)
        seq_length = self.window_size ** 2

        mask = ops.ones((seq_length, seq_length))
        mask = ops.expand_dims(mask, 0)
        x = super().call(x, mask)
        x = ops.reshape(x, (bz, x.shape[1], self.window_size, self.window_size, C))
        x = ops.reshape(x, (bz, int(np.sqrt(x.shape[1])), int(np.sqrt(x.shape[1])), 
                            self.window_size, self.window_size, C
                            ))
        
        x = self.merge_layer(x)
        x = ops.squeeze(x, (1, 2))
        return x

class SwineEncoderBlock(keras.Layer):
    def __init__(self, window_size, att_tp, **kwargs):
        super().__init__(**kwargs)
        self.att_tp = att_tp
        self.window_size = window_size

    def build(self, input_shape):
        self.ln1 = layers.LayerNormalization()
        self.ln2 = layers.LayerNormalization()
        
        self.att_fn = WMSA(self.window_size) if self.att_tp == "window" else SWMSA()
        
        self.residual1 = layers.Add()
        self.residual2 = layers.Add()
        
        self.mlp = MLP()
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):
        prev_x = x
        x = self.ln1(x)
        x = self.att_fn(x)
        x = self.residual1([x, prev_x])
        
        prev_x = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = self.residual2([x, prev_x])
        return x
    
class SwineEncoder(keras.Layer):
    def __init__(self, window_size=7, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size

    def build(self, input_shape):
        self.window_eb = SwineEncoderBlock(window_size=7, att_tp="window")
        self.shifted_window_eb = SwineEncoderBlock(window_size=7, att_tp="shifted_window")
        return super().build(input_shape)

    def call(self, x):
        x = self.window_eb(x)
        # x = self.shifted_window_eb(x)
        return x
    
class SwinBlock(keras.Layer):
    def __init__(self, patch_size, window_size=7, nblocks=2, 
                 merge=True, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.window_size = window_size
        self.merge = merge
        self.nblocks = nblocks

    def build(self, input_shape):
        bz, H, W, C = input_shape
        self.new_C = 2*C
        self.merge_op = (keras.Sequential(
            [PartitionMerging(), 
             layers.Reshape((H // 2, W//2, 4*C)),
             layers.Dense(C * 2), 
             ]) if self.merge 
            else keras.Sequential([layers.Identity(), 
                                   layers.Reshape((H , W, -1))
                                   ])
        )
        self.swin_encoder = keras.Sequential([SwineEncoder() for _ in range(self.nblocks // 2)])
        return super().build(input_shape)
    
    def call(self, x):
        bz, h, w, C = ops.shape(x)
        x = ops.expand_dims(ops.expand_dims(x, 3), 4)
        x = self.merge_op(x)
        x = self.swin_encoder(x)
        return x

class SwinTransformer(keras.Layer):
    def __init__(self, C,  name="swin_transformer", **kwargs):
        super().__init__(name=name, **kwargs)
        self.C = C

    def build(self, input_shape):
        _, H, W, c = input_shape
        self.rescaling = layers.Rescaling(
            1/255., name=f"{self.name}_rescaling")
        
        # Patch partition B, num_p, num_p, pz, pz, 2
        self.patch_partition = PartitionLayer(4)
        # Linear Embedding
        self.reshape = layers.Reshape((H // 4 , W // 4, -1))
        self.proj1 = layers.Dense(self.C)

        self.swin_block1 = SwinBlock(patch_size=4, window_size=7, merge=False)
        self.swin_block2 = SwinBlock(patch_size=8, window_size=7, merge=True)
        self.swin_block3 = SwinBlock(patch_size=16, window_size=7, nblocks=6)
        self.swin_block4 = SwinBlock(patch_size=32, window_size=7,)

        self.reshape_pool = layers.Reshape((H//32 * W//32, -1))
        return super().build(input_shape)

    def call(self, x):
        x = self.rescaling(x)
        x = self.patch_partition(x)
        x = self.reshape(x)
        x = self.proj1(x)
        
        # (None, npatch, npatch, C)  
        x = self.swin_block1(x)
        x = self.swin_block2(x)
        x = self.swin_block3(x)
        x = self.swin_block4(x)

        return x