import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
data = pd.read_csv('Data/Student_data.csv')

# Kiểm tra và xử lý dữ liệu bị thiếu
data.dropna(inplace=True)

# Tạo các đặc trưng mới
data['food_ratio'] = data['food_cost'] / data['total']
data['rent_ratio'] = data['rent'] / data['total']
data['other_cost_ratio'] = data['other_cost'] / data['total']

# Chia dữ liệu thành các biến độc lập (X) và biến phụ thuộc (y)
X = data[['food_cost', 'rent', 'other_cost']]
y = data[['food_ratio', 'rent_ratio', 'other_cost_ratio']]

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Tạo mô hình Random Forest
model = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=10, random_state=42)

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Nhập tổng tiền từ người dùng
while True:
    total_amount_str = input("Nhập tổng số tiền: ")
    total_amount_str = total_amount_str.replace(",", "").replace(" ", "")
    total_amount_str = total_amount_str.replace(".", "").replace(" ", "")
    if "tr" in total_amount_str:
        total_amount_str = total_amount_str.replace("tr", "")
        total_amount = float(total_amount_str) * 1000000
    else:
        try:
            total_amount = float(total_amount_str)
            break
        except ValueError:
            print("Vui lòng nhập số!")

# Tính giá trị trung bình của các cột other_cost và savings
mean_other_cost = data['other_cost'].mean()

# Chuẩn hóa dữ liệu đầu vào
input_data = scaler.transform([[total_amount, mean_other_cost, 0]])

# Dự đoán tỷ lệ phần trăm cho mỗi mục
predicted_ratios = model.predict(input_data)[0]

# Kiểm tra và điều chỉnh giá trị dự đoán
predicted_ratios = np.maximum(predicted_ratios, 0)  # Đảm bảo tất cả giá trị đều không âm
predicted_ratios /= np.sum(predicted_ratios)  # Chuẩn hóa lại tổng tỷ lệ bằng 1

# Tính số tiền cho mỗi mục
predicted_food = round(predicted_ratios[0] * total_amount / 1000) * 1000
predicted_rent = round(predicted_ratios[1] * total_amount / 1000) * 1000
predicted_other = round(predicted_ratios[2] * total_amount / 1000) * 1000

# In kết quả với định dạng số có dấu cách sau mỗi 3 chữ số
print("Phân bổ chi tiêu dự kiến:")
print("Ăn uống:", "{:,}".format(predicted_food))
print("Nhà ở:", "{:,}".format(predicted_rent))
print("Chi phí khác:", "{:,}".format(predicted_other))

# Kiểm tra lại tổng
total_predicted = predicted_food + predicted_rent + predicted_other 
print("Tổng chi tiêu dự đoán:", "{:,}".format(total_predicted))

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

print(f"Mô hình có khả năng dự đoán {r2*100:.2f}%")
print(f"Adjusted R²: {adjusted_r2:.2f}")
print("Các chỉ số đánh giá:")
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("Mean Absolute Error:", mae)

# Trực quan hóa kết quả dự đoán
# plt.scatter(y_test.values.flatten(), y_pred.flatten(), alpha=0.5)
# plt.plot([0, 1], [0, 1], '--r')
# plt.xlabel("Giá trị thực")
# plt.ylabel("Giá trị dự đoán")
# plt.title("Dự đoán so với giá trị thực")
# plt.show()