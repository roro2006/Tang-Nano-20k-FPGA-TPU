import unittest

from host import decode_q8_matrix, encode_q8_matrix
from model_export import embed_grid, full_inference, layer_norm_q8_8, positional_embedding
from q8_attention import attention_q8_8, matmul_q8_8, pack_frame, unpack_frame


class Q8AttentionTests(unittest.TestCase):
    def test_matmul(self):
        self.assertEqual(matmul_q8_8([[256, 128]], [[128], [256]]), [[256]])

    def test_attention_shape_and_determinism(self):
        x = [[256 if i == j else 0 for j in range(4)] for i in range(4)]
        eye = [[256 if i == j else 0 for j in range(4)] for i in range(4)]
        result = attention_q8_8(x, eye, eye, eye)
        self.assertEqual((len(result), len(result[0])), (4, 4))
        self.assertEqual(result, attention_q8_8(x, eye, eye, eye))

    def test_packet_round_trip_and_corruption(self):
        frame = pack_frame(b"\x01\x02\x03")
        self.assertEqual(unpack_frame(frame), b"\x01\x02\x03")
        with self.assertRaises(ValueError):
            unpack_frame(frame[:-1] + b"\x01")

    def test_host_matrix_codec(self):
        matrix = [[-256, 0], [128, 32767]]
        self.assertEqual(decode_q8_matrix(encode_q8_matrix(matrix), 2, 2), matrix)

    def test_layer_norm_shape(self):
        result = layer_norm_q8_8([[256, 512, 768, 1024]])
        self.assertEqual((len(result), len(result[0])), (1, 4))

    def test_embedding_and_full_inference_identity(self):
        grid = [[0] * 16 for _ in range(16)]
        identity = [[256 if row == col else 0 for col in range(64)] for row in range(64)]
        output = full_inference(
            grid,
            [{"q": identity, "k": identity, "v": identity, "o": identity}] * 2,
            [(identity, identity)] * 2,
            [[256 if row == col else 0 for col in range(16)] for row in range(64)],
        )
        self.assertEqual((len(embed_grid(grid)), len(positional_embedding())), (16, 16))
        self.assertEqual((len(output), len(output[0])), (16, 16))


if __name__ == "__main__":
    unittest.main()
