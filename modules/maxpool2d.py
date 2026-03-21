from modules.layer import Layer
#from cython_modules.maxpool2d import maxpool_forward_cython
import numpy as np

class MaxPool2D(Layer):
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward_original(self, input, training=True):  # input: np.ndarray of shape [B, C, H, W]
        self.input = input
        B, C, H, W = input.shape
        KH, KW = self.kernel_size, self.kernel_size
        SH, SW = self.stride, self.stride

        out_h = (H - KH) // SH + 1
        out_w = (W - KW) // SW + 1

        self.max_indices = np.zeros((B, C, out_h, out_w, 2), dtype=int)
        output = np.zeros((B, C, out_h, out_w),dtype=input.dtype)

        for b in range(B):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * SH
                        h_end = h_start + KH
                        w_start = j * SW
                        w_end = w_start + KW

                        window = input[b, c, h_start:h_end, w_start:w_end]
                        max_idx = np.unravel_index(np.argmax(window), window.shape)
                        max_val = window[max_idx]

                        output[b, c, i, j] = max_val
                        self.max_indices[b, c, i, j] = (h_start + max_idx[0], w_start + max_idx[1])

        return output
    
    ## ----- Generado con IA
    def forward(self, input, training=True):
        self.input = input
        B, C, H, W = input.shape
        KH, KW = self.kernel_size, self.kernel_size
        SH, SW = self.stride, self.stride
    
        out_h = (H - KH) // SH + 1
        out_w = (W - KW) // SW + 1
    
        # Crear vista de todas las ventanas simultáneamente
        shape = (B, C, out_h, out_w, KH, KW)
        strides = (
            input.strides[0],
            input.strides[1],
            input.strides[2] * SH,
            input.strides[3] * SW,
            input.strides[2],
            input.strides[3],
        )
        windows = np.lib.stride_tricks.as_strided(input, shape=shape, strides=strides)
    
        # Máximo sobre las dos últimas dimensiones (KH, KW) en una sola operación
        output = windows.max(axis=(4, 5))
    
        # Índices del máximo (aplanados dentro de cada ventana)
        flat_idx = windows.reshape(B, C, out_h, out_w, -1).argmax(axis=4)
    
        # Convertir índice plano → coordenadas (r, s) globales
        local_r, local_s = np.unravel_index(flat_idx, (KH, KW))
    
        # Sumar el offset de cada ventana para obtener coordenadas globales
        row_offsets = np.arange(out_h)[None, None, :, None] * SH
        col_offsets = np.arange(out_w)[None, None, None, :] * SW
    
        self.max_indices = np.stack([
            local_r + row_offsets,
            local_s + col_offsets
        ], axis=-1)
    
        return output
## ---------- Fin generado con IA

    def backward_oirignal(self, grad_output, learning_rate=None):
        B, C, H, W = self.input.shape
        grad_input = np.zeros_like(self.input, dtype=grad_output.dtype)
        out_h, out_w = grad_output.shape[2], grad_output.shape[3]

        for b in range(B):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        r, s = self.max_indices[b, c, i, j]
                        grad_input[b, c, r, s] += grad_output[b, c, i, j]

        return grad_input
    ## ----- Generado con IA
    def backward(self, grad_output):
        B, C, out_h, out_w = grad_output.shape
        grad_input = np.zeros_like(self.input)

        # 1. Creamos índices para las dimensiones Batch y Canal
        # Usamos np.arange y broadcasting para que coincidan con la forma de grad_output
        b_idx = np.arange(B).reshape(B, 1, 1, 1)
        c_idx = np.arange(C).reshape(1, C, 1, 1)
        
        # 2. Extraemos las coordenadas r y s guardadas en el forward
        # Asumiendo que max_indices tiene forma (B, C, out_h, out_w, 2)
        r_idx = self.max_indices[:, :, :, :, 0]
        s_idx = self.max_indices[:, :, :, :, 1]

        # 3. Dispersión del gradiente (Scatter)
        # NumPy permite usar arrays de índices para asignar valores en bloque
        np.add.at(grad_input, (b_idx, c_idx, r_idx, s_idx), grad_output)

        return grad_input
    ## ---------- Fin generado con IA