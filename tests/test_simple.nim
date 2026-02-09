include nuwa_sdk
import nimpy

let np = pyImport("numpy")
let arr = np.array([1, 2, 3], dtype="int64")

echo "Creating numpy array wrapper..."
let npArr = asNumpyArray(arr, int64)
echo "Length: ", npArr.len
echo "Success!"
