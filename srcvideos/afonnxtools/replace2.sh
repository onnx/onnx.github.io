#find . -type f -name "*.html" -exec sed -i 's+"/theme+"https://andife.github.io/onnxvideo.github.io/theme+g' {} +

#find . -type f -name "*.html" -exec sed -i 's+"theme+"https://andife.github.io/onnxvideo.github.io/theme+g' {} +

find . -type f -name "*.html" -exec sed -i 's+<a href="/onnx-community-day-2022_06/+<a href="https://andife.github.io/onnx.github.io/videos/onnx-community-day-2022_06/+g' {} +

find . -type f -name "*.html" -exec sed -i 's+<a href="/onnx-community-day-2021_03/+<a href="https://andife.github.io/onnx.github.io/videos/onnx-community-day-2021_03/+g' {} +
