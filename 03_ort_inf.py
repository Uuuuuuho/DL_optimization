import onnxruntime as rt
import numpy as np
import os
import pdb
from profiler import CudaProfiler

prof = CudaProfiler()

def generate_dummy_input(shape: tuple, dtype: np.dtype) -> np.ndarray:
    """
    주어진 shape와 dtype에 맞는 더미 입력 텐서를 생성합니다.
    int64 타입의 경우 특정 범위의 정수 값을 생성합니다.
    """
    if dtype == np.float32 or dtype == np.float16 or dtype == np.float64:
        return np.random.uniform(low=-1, high=1, size=shape).astype(dtype) # 0~1 사이의 임의 값
        # return np.random.rand(*shape).astype(dtype) # 0~1 사이의 임의 값
    elif dtype == np.int64 or dtype == np.int32:
        # int64의 경우, 일반적으로 ID나 인덱스 등 특정 범위의 값을 가짐
        return np.random.randint(0, 10, size=shape, dtype=dtype) # 0~9 사이의 임의 정수
    elif dtype == np.bool_:
        # int64의 경우, 일반적으로 ID나 인덱스 등 특정 범위의 값을 가짐
        return np.random.randint(0, 2, size=shape, dtype=dtype).astype(np.bool_) # 0~9 사이의 임의 정수
    else:
        raise ValueError(f"지원하지 않는 데이터 타입: {dtype}")

def get_onnx_input_output_info(sess: rt.InferenceSession):
    """
    ONNX InferenceSession으로부터 입력 및 출력 텐서의 이름, 형태, 타입을 추출합니다.
    """
    input_info = {}
    for _input in sess.get_inputs():
        input_info[_input.name] = {
            'shape': _input.shape,
            'dtype': np.dtype(_input.type.replace('tensor(', '').replace(')', ''))
        }

    output_info = {}
    for _output in sess.get_outputs():
        output_info[_output.name] = {
            'shape': _output.shape,
            'dtype': np.dtype(_output.type.replace('tensor(', '').replace(')', ''))
        }
    return input_info, output_info

def load_input_numpy(input_tensor, dtype):
    file_path = ""
    tensor_dir = "/home/uho/workspace/ua-prof/onnx_tensors"
    if input_tensor == "input":
        file_path = tensor_dir + "/ort_inputs_input.npy"
    if input_tensor == "a_idcs":
        file_path = tensor_dir + "/ort_inputs_a_idcs.npy"
    if input_tensor == "onnx::MatMul_2":
        file_path = tensor_dir + "/ort_inputs_onnx_2.npy"
    if input_tensor == "onnx::Gather_3":
        file_path = tensor_dir + "/ort_inputs_onnx_3.npy"
    if input_tensor == "onnx::Transpose_4":
        file_path = tensor_dir + "/ort_inputs_onnx_4.npy"
    
    try:
        with open(file_path, 'rb') as f:
            tensor = np.load(f).astype(dtype)
            if isinstance(tensor, np.ndarray):
                return tensor
    except FileNotFoundError:
        raise FileNotFoundError(f"File does not exist")
    except Exception as e:
        raise Exception(f"PKL file load error occurred: {e}")
    
def save_tensor_bin(data, file_path, name):
    if name == "input":
        data = np.asarray(data, dtype=np.float32)
    if name == "a_idcs":
        data = np.asarray(data, dtype=np.int64)
    if name == "onnx::MatMul_2":
        data = np.asarray(data, dtype=np.float32)
    if name == "onnx::Gather_3":
        data = np.asarray(data, dtype=np.int64)
    if name == "onnx::Transpose_4":
        data = np.asarray(data, dtype=np.float32)
    data.tofile(file_path)
    
def run_onnx_inference(
    onnx_model_path: str,
    input_specs: dict, # {'input_name': (shape, dtype)}
    output_names: list,
    execution_provider: str = 'CUDAExecutionProvider',
    save_tensors: bool = False,
    save_dir: str = "onnx_tensors",
    ep_options = {}
):
    """
    ONNX 모델을 실행하고 입력/출력 텐서를 저장(선택 사항)합니다.

    Args:
        onnx_model_path (str): ONNX 모델 파일의 경로.
        input_specs (dict): 입력 텐서의 이름, 형태, 타입을 담은 딕셔너리.
                            예: {'input1': ((50,14,48), np.float32), ...}
        output_names (list): 모델의 출력 텐서 이름 리스트.
        execution_provider (str): ONNX Runtime 실행 공급자 (예: 'CPUExecutionProvider', 'CUDAExecutionProvider').
        save_tensors (bool): 입력 및 출력 텐서를 파일로 저장할지 여부.
        save_dir (str): 텐서 파일을 저장할 디렉토리 경로.
    """

    print(f"\n--- ONNX 모델 테스트 시작: {onnx_model_path} ---")
    print(f"  실행 공급자: {execution_provider}")

    try:
        # ONNX Runtime 세션 생성
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_DISABLE_ALL
        # 세션 옵션 추가 가능 (예: sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL)

        sess = rt.InferenceSession(onnx_model_path, sess_options, providers=[(execution_provider, ep_options)])
        print(f"'{onnx_model_path}' 모델 로드 성공.")

        # 모델의 실제 입출력 정보 확인 (sanity check)
        model_input_info, model_output_info = get_onnx_input_output_info(sess)
        print("\n  모델 로드 후 확인된 입력 정보:")
        for name, info in model_input_info.items():
            print(f"    - {name}: Shape={info['shape']}, Dtype={info['dtype']}")
        print("\n  모델 로드 후 확인된 출력 정보:")
        for name, info in model_output_info.items():
            print(f"    - {name}: Shape={info['shape']}, Dtype={info['dtype']}")

        # 입력 데이터 생성
        inputs = {}
        for name, spec in input_specs.items():
            shape, dtype = spec
            # inputs[name] = load_input_numpy(name, dtype)
            # print(f"  더미 입력 '{name}' 로딩 완료. Shape: {inputs[name].shape}, Dtype: {inputs[name].dtype}")
            inputs[name] = generate_dummy_input(shape, dtype)
            print(f"  더미 입력 '{name}' 생성 완료. Shape: {inputs[name].shape}, Dtype: {inputs[name].dtype}")

        # # 입력 텐서 저장
        # if save_tensors:
        #     os.makedirs(save_dir, exist_ok=True)
        #     for name, data in inputs.items():
        #         if name.startswith("onnx::"):
        #             name = name.replace("onnx::", "onnx_", 1)
        #         input_file_path = os.path.join(save_dir, f"{name}_input.bin")
        #         # np.save(input_file_path, data)
        #         save_tensor_bin(data, input_file_path, name)
        #         # data = np.asarray(data, dtype=np.float32)
        #         # data.tofile(input_file_path)
        #         print(f"  입력 텐서 '{name}'를 '{input_file_path}'에 저장했습니다.")

        # 모델 실행
        num_iter = 300
        print("  모델 실행 중...")
        for _ in range(num_iter):
            prof.start("ORT_run()")
            outputs = sess.run(output_names, inputs)
            prof.stop("ORT_run()")
            prof.synchronize_all(0)
        print("  모델 실행 완료.")

        # 출력 텐서 저장 (선택 사항)
        if save_tensors:
            for i, output_tensor in enumerate(outputs):
                name = output_names[i]
                output_file_path = os.path.join(save_dir, f"ort_outputs_{name}.npy")
                np.save(output_file_path, output_tensor)
                print(f"  출력 텐서 '{name}'를 '{output_file_path}'에 저장했습니다. (Shape: {output_tensor.shape}, Dtype: {output_tensor.dtype})")
            print(f"  모든 입력 및 출력 텐서가 '{save_dir}'에 저장되었습니다.")
        
        print(f"\n--- ONNX 모델 테스트 완료: {onnx_model_path} ---")
        return outputs

    except Exception as e:
        print(f"\n!!! ONNX 모델 테스트 중 오류 발생: {onnx_model_path}")
        print(f"오류 내용: {e}")
        return None

# --- 메인 실행 부분 ---
if __name__ == "__main__":
    # --- 0. ONNX 모델 경로 설정 ---
    # 여기에 여러분의 FP32 ONNX 모델 파일 경로를 지정하세요.
    # fp32_onnx_model_path = "./onnx_out/parallel_matmul_fused.onnx"
    # fp32_onnx_model_path = "./onnx_out/parallel_matmul.onnx"
    fp32_onnx_model_path = "./onnx_out/SymmetricFT.onnx"
    
    # 여기에 여러분의 FP16 ONNX 모델 파일 경로를 지정하세요.
    # FP16 모델이 없다면, 이 변수를 주석 처리하거나 빈 문자열로 두면 FP16 테스트를 건너뜁니다.
    # fp16_onnx_model_path = "/home/uho/workspace/ua-prof/models/[0.5.3]mo_net_fp16.onnx"

    # ep = "TensorrtExecutionProvider"
    ep = "CUDAExecutionProvider"

    # --- 1. 입력/출력 텐서 정보 정의 ---
    # input_specs는 모델의 실제 입력 이름과 그에 해당하는 형태, 데이터 타입을 정확히 명시해야 합니다.
    # onnxsim을 사용하기 전에 ONNX 모델의 정확한 입출력 이름을 Netron 등으로 확인하는 것이 좋습니다.

    # input_specs = {
    #     "X": ((1, 8192), np.float32),
    # }
    # output_names = [
    #     "Y1_fused", "Y2_fused"
    # ]
    # output_names = [
    #     "Y1", "Y2"
    # ]
    input_specs = {
        "tokens": ((300, 128), np.float32),
        "rpe": ((300, 300, 128), np.float32),
        "rpes": ((300, 300), np.bool_),
    }
    output_names = [
        "out"
    ]

    # --- 2. ONNX Runtime 테스트 ---
    # FP32 모델 테스트
    
    if os.path.exists(fp32_onnx_model_path):
        print("\n=== FP32 모델 테스트 시작 ===")
        # CUDA 실행 공급자 (GPU가 있는 경우)
        # onnxruntime-gpu가 설치되어 있어야 합니다. (pip install onnxruntime-gpu)
        if ep == "CUDAExecutionProvider":
            print("\n--- CUDA Execution Provider ---")
            run_onnx_inference(
                onnx_model_path=fp32_onnx_model_path,
                input_specs=input_specs,
                output_names=output_names,
                execution_provider=ep,
                save_tensors=False # GPU 테스트에서는 텐서 저장은 비활성화
            )
        elif ep == "TensorrtExecutionProvider":
            print("\n--- CUDA Execution Provider ---")
            run_onnx_inference(
                onnx_model_path=fp32_onnx_model_path,
                input_specs=input_specs,
                output_names=output_names,
                execution_provider=ep,
                save_tensors=False, # GPU 테스트에서는 텐서 저장은 비활성화
            )
        else:
            print("\nCUDA Execution Provider를 사용할 수 없습니다. (GPU 또는 onnxruntime-gpu 미설치)")
    else:
        print(f"\n경고: FP32 모델 '{fp32_onnx_model_path}'을(를) 찾을 수 없습니다. FP32 테스트를 건너뜝니다.")


    # # FP16 모델 테스트
    # trt_ep_options = {
    #     "trt_fp16_enable": True
    # }
    # if os.path.exists(fp16_onnx_model_path):
    #     print("\n=== FP16 모델 테스트 시작 ===")
    #     # FP16 모델이므로 입력 데이터도 float16으로 변경되어야 합니다.
    #     fp16_input_specs = {
    #         "input": ((50, 14, 48), np.float16),
    #         "a_idcs": ((50,), np.int64),
    #         "onnx::MatMul_2": ((250, 10, 17), np.float16),
    #         "onnx::Gather_3": ((250,), np.int64),
    #         "onnx::Transpose_4": ((5, 300, 300), np.float16),
    #     }

    #     # CUDA 실행 공급자 (GPU가 있는 경우, FP16 연산은 주로 GPU에서 효율적)
    #     if ep == "CUDAExecutionProvider":
    #         print("\n--- CUDA Execution Provider (for FP16 model) ---")
    #         run_onnx_inference(
    #             onnx_model_path=fp16_onnx_model_path,
    #             input_specs=fp16_input_specs,
    #             output_names=output_names,
    #             execution_provider='TensorrtExecutionProvider',
    #             save_tensors=True,
    #             ep_options=trt_ep_options
    #         )
    #     elif ep == "TensorrtExecutionProvider":
    #         print("\n--- TensorRT Execution Provider (for FP16 model) ---")
    #         run_onnx_inference(
    #             onnx_model_path=fp16_onnx_model_path,
    #             input_specs=fp16_input_specs,
    #             output_names=output_names,
    #             execution_provider='TensorrtExecutionProvider',
    #             save_tensors=True,
    #             ep_options=trt_ep_options
    #         )
    #     else:
    #         print("\nCUDA Execution Provider를 사용할 수 없습니다. (GPU 또는 onnxruntime-gpu 미설치)")
    # else:
    #     print(f"\n경고: FP16 모델 '{fp16_onnx_model_path}'을(를) 찾을 수 없습니다. FP16 테스트를 건너뜝니다.")