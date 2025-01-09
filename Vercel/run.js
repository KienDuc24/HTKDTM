const { spawn } = require('child_process');
const { google } = require('googleapis');
require('dotenv').config();
const KEYFILE = process.env.KEYFILE;
const SPREADSHEETID = process.env.SPREADSHEETID;

async function getGoogleSheetData() {
  const auth = new google.auth.GoogleAuth({
    keyFile: KEYFILE,
    scopes: ['https://www.googleapis.com/auth/spreadsheets'], // Quyền đọc và ghi
  });

  const sheets = google.sheets({ version: 'v4', auth });

  const spreadsheetId = SPREADSHEETID;
  const range = 'Sheet2!G7'; // Phạm vi bạn muốn lấy dữ liệu

  try {
    const response = await sheets.spreadsheets.values.get({
      spreadsheetId,
      range,
    });

    const rows = response.data.values;

    if (Array.isArray(rows) && rows.length > 0 && rows[0].length > 0) {
      const value = parseFloat(rows[0][0]);
      if (!isNaN(value)) {
        console.log('Tổng tiền:', value);
        runPythonScript(value);
      } else {
        console.log('Giá trị không hợp lệ trong ô K1.');
      }
    } else {
      console.log('Không tìm thấy dữ liệu trong phạm vi ô được chỉ định.');
    }
  } catch (error) {
    console.error('Lỗi khi lấy dữ liệu từ Google Sheets:', error);
  }
}


function runPythonScript(total) {
    return new Promise((resolve, reject) => {
      const pythonProcess = spawn('python', ['ML/Employee_LinearRegression.py', total]);
  
      // Xử lý dữ liệu trả về từ Python
      let result = '';
      pythonProcess.stdout.on('data', (data) => {
        console.log(`${data.toString()}`);
        result += data.toString().trim() + '\n'; // Lưu tất cả dữ liệu
      });
  
    //   Xử lý lỗi nếu có
      pythonProcess.stderr.on('data', (data) => {
        console.error(`Lỗi từ Python: ${data.toString()}`);
      });
  
      // Xử lý khi quá trình thực thi hoàn tất
      pythonProcess.on('close', (code) => {
        if (code === 0) {
            // Chuyển chuỗi thành mảng tách theo dòng
            const resultArray = result.split('\n').map(item => item.trim());
            writeToSheet(resultArray).then(resolve).catch(reject);
          } else {
            reject(new Error(`Quá trình Python kết thúc với mã: ${code}`));
          }
      });
    });
  }

async function writeToSheet(result) {
    const auth = new google.auth.GoogleAuth({
      keyFile: KEYFILE,
      scopes: ['https://www.googleapis.com/auth/spreadsheets'],
    });
  
    const sheets = google.sheets({ version: 'v4', auth });
  
    const spreadsheetId = SPREADSHEETID;
    const range = 'Sheet2!H2:H6'; // Vị trí mà bạn muốn ghi kết quả
  
    const resource = {
        values: result.map(value => [value.replace(/,/g, '')]), // Loại bỏ dấu phẩy
    };
  
    await sheets.spreadsheets.values.update({
      spreadsheetId,
      range,
      valueInputOption: 'RAW',
      resource,
    });
  
    console.log('Dữ liệu đã được ghi vào Google Sheets.');
  }

// Gọi hàm để lấy dữ liệu từ Google Sheets
getGoogleSheetData().catch(console.error);