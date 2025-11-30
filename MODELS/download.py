import os
import subprocess
import zipfile
import shutil

def download_mnist_from_kaggle():
    """
    Tải MNIST dataset từ Kaggle và giải nén vào thư mục data.
    
    Yêu cầu:
    - Đã cài đặt Kaggle API: pip install kaggle
    - Đã cấu hình Kaggle credentials (~/.kaggle/kaggle.json)
    """
    
    # Đường dẫn thư mục hiện tại
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir,'data')
    download_path = os.path.join(data_dir, 'mnist-dataset.zip')
    
    print("=" * 60)
    print("Tải MNIST Dataset từ Kaggle")
    print("=" * 60)
    
    # Kiểm tra xem đã có dataset chưa
    required_files = [
        'train-images-idx3-ubyte',
        'train-labels-idx1-ubyte',
        't10k-images-idx3-ubyte',
        't10k-labels-idx1-ubyte'
    ]
    
    all_exist = all(os.path.exists(os.path.join(data_dir, f)) for f in required_files)
    if all_exist:
        print("✓ Dataset đã tồn tại, bỏ qua tải xuống")
        return
    
    try:
        # Tải dataset từ Kaggle
        print("\n1. Đang tải dataset từ Kaggle...")
        subprocess.run([
            'kaggle', 'datasets', 'download', '-d', 'hojjatk/mnist-dataset',
            '-p', data_dir
        ], check=True)
        
        print("✓ Tải xuống thành công!")
        
        # Giải nén file
        print("\n2. Đang giải nén...")
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        print("✓ Giải nén thành công!")
        
        # Xóa file zip
        print("\n3. Dọn dẹp...")
        os.remove(download_path)
        print("✓ Đã xóa file zip")
        
        # Xóa các thư mục trùng lặp (chỉ giữ file ở root của data/)
        print("\n4. Xóa các thư mục trùng lặp...")
        duplicate_folders = [
            't10k-images-idx3-ubyte',
            't10k-labels-idx1-ubyte',
            'train-images-idx3-ubyte',
            'train-labels-idx1-ubyte'
        ]
        
        for folder in duplicate_folders:
            folder_path = os.path.join(data_dir, folder)
            if os.path.isdir(folder_path):
                shutil.rmtree(folder_path)
                print(f"   ✓ Đã xóa folder: {folder}")
        
        # Kiểm tra files
        print("\n5. Kiểm tra files đã tải:")
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            if os.path.isfile(filepath) and filename.endswith(('-ubyte', '.gz', '.csv')):
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"   ✓ {filename} ({size_mb:.2f} MB)")
        
        print("\n" + "=" * 60)
        print("✓ Hoàn tất! Dataset đã được tải về thư mục data/")
        print("=" * 60)
        
    except subprocess.CalledProcessError:
        print("\n✗ Lỗi: Không thể tải từ Kaggle")
        print("\nHướng dẫn cài đặt Kaggle API:")
        print("1. Cài đặt: pip install kaggle")
        print("2. Tạo API token tại: https://www.kaggle.com/settings")
        print("3. Tải file kaggle.json và đặt vào:")
        print("   - Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json")
        print("   - Linux/Mac: ~/.kaggle/kaggle.json")
        print("4. Chạy lại script này")
        
    except Exception as e:
        print(f"\n✗ Lỗi: {e}")


def download_with_curl():
    """
    Tải MNIST dataset bằng curl (backup method).
    Sử dụng khi không có Kaggle API.
    """
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir,'data')
    download_path = os.path.join(data_dir, 'mnist-dataset.zip')
    
    print("=" * 60)
    print("Tải MNIST Dataset bằng curl")
    print("=" * 60)
    
    try:
        print("\nĐang tải dataset...")
        
        # Sử dụng curl để tải
        curl_command = [
            'curl', '-L', '-o', download_path,
            'https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset'
        ]
        
        subprocess.run(curl_command, check=True)
        
        print("✓ Tải xuống thành công!")
        
        # Giải nén
        print("\nĐang giải nén...")
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        print("✓ Giải nén thành công!")
        
        # Xóa file zip
        os.remove(download_path)
        
        # Xóa các thư mục trùng lặp
        duplicate_folders = [
            't10k-images-idx3-ubyte',
            't10k-labels-idx1-ubyte',
            'train-images-idx3-ubyte',
            'train-labels-idx1-ubyte'
        ]
        
        for folder in duplicate_folders:
            folder_path = os.path.join(data_dir, folder)
            if os.path.isdir(folder_path):
                shutil.rmtree(folder_path)
        
        print("✓ Hoàn tất!")
        
    except Exception as e:
        print(f"\n✗ Lỗi: {e}")
        print("\nLưu ý: Phương thức này yêu cầu Kaggle authentication.")
        print("Vui lòng sử dụng method download_mnist_from_kaggle() với Kaggle API.")


if __name__ == '__main__':
    print("\nChọn phương thức tải:")
    print("1. Kaggle API (khuyên dùng)")
    print("2. curl (cần authentication)")
    
    choice = input("\nNhập lựa chọn (1/2): ").strip()
    
    if choice == '1':
        download_mnist_from_kaggle()
    elif choice == '2':
        download_with_curl()
    else:
        print("Lựa chọn không hợp lệ. Sử dụng Kaggle API...")
        download_mnist_from_kaggle()
