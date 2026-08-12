# Tài liệu bàn giao Vi-FDB v1.0

## Liên kết

- Bộ dữ liệu: <https://huggingface.co/datasets/tuanamz/vi-fdb-v1>
- Bộ công cụ đánh giá: <https://github.com/tuanad121/Full-Duplex-Bench/tree/main/vi_fdb_harness>
- Output tham chiếu của GPT-Realtime: <https://huggingface.co/datasets/tuanamz/vi-fdb-v1-gpt-realtime>
- Trình duyệt kết quả tương tác: <https://huggingface.co/spaces/tuanamz/vi-fdb-v1-explorer>

Bộ dữ liệu trên Hugging Face được công khai, không yêu cầu đăng nhập hay xin
quyền truy cập. Bộ dữ liệu tuân theo giấy phép CC BY-NC 4.0 của
Full-Duplex-Bench gốc. Vi-FDB được công bố độc lập và không phải bộ dữ liệu chính
thức của VinFast.

## Benchmark này đo lường điều gì?

Vi-FDB đánh giá hệ thống hội thoại giọng nói full-duplex bằng tiếng Việt, tức hệ
thống có khả năng nghe và nói đồng thời. Benchmark kiểm tra khi nào hệ thống nên
chờ, bắt đầu nói, dừng lại khi người dùng thực sự ngắt lời, hoặc tiếp tục nói khi
có tín hiệu chồng lấn không gây gián đoạn. Bộ dữ liệu gồm 400 tình huống tiếng
Việt tổng hợp thuộc bảy tác vụ:

| Tác vụ | Số mẫu | Hành vi mong đợi |
|---|---:|---|
| Backchannel | 50 | Phát tín hiệu phản hồi ngắn tự nhiên khi đang lắng nghe, nếu phù hợp |
| Xử lý khoảng dừng | 50 | Không giành lượt nói khi người dùng chỉ tạm dừng giữa câu |
| Chuyển lượt mượt mà | 50 | Trả lời sau khi người dùng đã nói xong |
| Tiếng nói nền | 50 | Bỏ qua lời nói không hướng đến hệ thống |
| Người dùng nói với người khác | 50 | Bỏ qua câu nói phụ hướng đến một người khác |
| Backchannel của người dùng | 50 | Tiếp tục nói khi người dùng chỉ phát tín hiệu phản hồi ngắn, không có ý ngắt lời |
| Người dùng ngắt lời | 100 | Dừng/nhường lượt và làm theo yêu cầu đã thay đổi; gồm mẫu chuẩn và mẫu có đối chứng ghép cặp |

Gói dữ liệu gồm hai tập con:

- `data/pilot_160`: đã chạy đầy đủ quy trình đầu-cuối với GPT-Realtime, ASR tiếng
  Việt, chấm tự động và hiệu chỉnh bởi người Việt bản ngữ.
- `data/expansion_240`: đã hoàn tất QC cấu trúc/audio, chạy GPT-Realtime,
  PhoWhisper timestamp mức từ, Silero VAD và các metric FDB gốc phù hợp.

Một submission đầy đủ dùng cả 400 mẫu event. 200 clean control là điều kiện đối
chứng ghép cặp, không phải các mẫu benchmark headline bổ sung.

## Tải dữ liệu

Cài Hugging Face CLI, sau đó tải bộ dữ liệu công khai:

```bash
hf download tuanamz/vi-fdb-v1 \
  --repo-type dataset \
  --local-dir vi-fdb-v1
```

Clone riêng bộ công cụ đánh giá:

```bash
git clone https://github.com/tuanad121/Full-Duplex-Bench.git
cd Full-Duplex-Bench/vi_fdb_harness
uv sync
```

Kiểm tra tập pilot đã được đánh giá đầy đủ:

```bash
uv run python harness.py validate-dataset \
  --dataset-root /absolute/path/to/vi-fdb-v1/data/pilot_160 \
  --profile original-160
```

Kết quả mong đợi: `ok: true`, tổng cộng 160 mẫu và 20 mẫu cho mỗi nhóm tác vụ/
phiên bản nguồn. Trong taxonomy của bản phát hành, hai biến thể ngắt lời được gộp
thành tác vụ `user_interruption` gồm 100 mẫu; harness vẫn giữ thông tin phương
pháp nguồn nội bộ để tính điểm tương thích.

Kiểm tra riêng tập mở rộng:

```bash
uv run python harness.py validate-dataset \
  --dataset-root /absolute/path/to/vi-fdb-v1/data/expansion_240 \
  --profile expansion-240
```

## Cấu trúc bộ dữ liệu

Mỗi tập con có một tệp `manifest.json`. Mỗi bản ghi trong manifest trỏ đến audio
và metadata của một tình huống:

```json
{
  "task": "background_speech",
  "id": "000001",
  "input": "background_speech/000001/input.wav",
  "clean_input": "background_speech/000001/clean_input.wav",
  "metadata": "background_speech/000001/metadata.json",
  "dataset_version": "1.0"
}
```

Các tệp quan trọng:

- `input.wav`: điều kiện có sự kiện, được truyền vào hệ thống cần đánh giá.
- `clean_input.wav`: mẫu đối chứng tương ứng, trong đó phần chồng lấn được thay
  bằng khoảng lặng; chỉ có ở các tình huống phù hợp.
- `metadata.json`: tác vụ, văn bản lời nói chính/sự kiện, khoảng thời gian sự
  kiện, thông tin người nói/ngữ cảnh và annotation dùng sau inference.
- `source_streams/`: các luồng audio chưa trộn, được giữ lại để kiểm tra.
- `index.html` hoặc `vibe_check.html`: trang thuận tiện để duyệt bộ dữ liệu.

Vai trò của sự kiện, văn bản nguồn và timestamp sự kiện đã gán nhãn là ground
truth. **Tuyệt đối không đưa các thông tin này cho hệ thống được đánh giá.** Chỉ
được dùng chúng sau inference trong bước đánh giá và kiểm tra thủ công. Một số
tệp JSON vẫn giữ trường `expected_action` cũ từ quá trình xây dựng dữ liệu cho
dialogue manager; trường này không thuộc cơ chế chấm điểm Vi-FDB và phải được bỏ
qua.

## Chạy một mô hình

Để smoke test, hãy bắt đầu bằng một cặp event/clean. Adapter có sẵn hỗ trợ OpenAI
Realtime:

```bash
export OPENAI_API_KEY=...

uv run python harness.py run-openai \
  --dataset-root /absolute/path/to/vi-fdb-v1/data/pilot_160 \
  --run-root ../outputs/vi_fdb_v1_0/my_model \
  --node-cli ../v1_v1.5/model_inference/gpt-realtime/cli.js \
  --condition both --jobs 1 --limit 1
```

Cài transport JavaScript một lần trước khi dùng adapter này:

```bash
cd ../v1_v1.5/model_inference/gpt-realtime
npm install
```

Đối với hệ thống speech-to-speech khác, hãy viết adapter tuân theo cùng output
contract. Yêu cầu cốt lõi là toàn bộ quá trình phải dùng chung một đồng hồ đơn
điệu:

1. Hoàn tất khởi tạo mô hình/session trước khi bắt đầu đồng hồ benchmark.
2. Đặt `t=0` ngay trước khi gửi frame PCM đầu tiên của input.
3. Truyền input theo timestamp tuyệt đối của audio, không gửi toàn bộ tệp nhanh
   nhất có thể.
4. Đặt các delta audio của trợ lý trên cùng đồng hồ và tính đúng thời gian phát
   thành tiếng, hủy phản hồi và cắt audio.
5. Lưu `output.wav` và `output_timing.json`; với mẫu ghép cặp, lưu thêm
   `clean_output.wav` và `clean_output_timing.json`.
6. Timeline output phải dài ít nhất bằng timeline input để vẫn quan sát được
   khoảng lặng và trường hợp hệ thống không phản hồi.

Kiểm tra một lượt chạy đã hoàn tất:

```bash
uv run python harness.py validate-run \
  --dataset-root /absolute/path/to/vi-fdb-v1/data/pilot_160 \
  --run-root ../outputs/vi_fdb_v1_0/my_model
```

Harness hỗ trợ tiếp tục lượt chạy bị gián đoạn. Không dùng `--overwrite` trừ khi
chủ động muốn tạo lại các mẫu đã hoàn tất.

## Nhận dạng tiếng nói, chấm điểm và duyệt kết quả

Cài các nhóm dependency tùy chọn cho ASR tiếng Việt và chấm tự động:

```bash
uv sync --group asr
uv sync --group judge
```

Chuyển lời output bằng PhoWhisper với timestamp mức từ, sau đó gắn biên tiếng
nói quan sát trực tiếp từ audio bằng Silero VAD:

```bash
uv run python transcribe.py \
  --root ../outputs/vi_fdb_v1_0/my_model \
  --backend phowhisper

uv run python align_phowhisper_vad.py \
  --root ../outputs/vi_fdb_v1_0/my_model
```

Chạy bộ chấm ẩn thông tin, có xét vai trò hội thoại. Bộ chấm nhận định nghĩa tác
vụ, kết quả ASR, bằng chứng timing và annotation tham chiếu sau inference; bộ chấm
không nhận token hành động của dialogue manager:

```bash
export OPENAI_API_KEY=...

uv run python judge.py \
  --root ../outputs/vi_fdb_v1_0/my_model \
  --asr-backend phowhisper_vad
```

Tạo trang duyệt audio/transcript được đồng bộ theo thời gian:

```bash
uv run python report.py \
  --run-root ../outputs/vi_fdb_v1_0/my_model \
  --asr-backend phowhisper_vad
```

Cần kiểm tra thủ công các quyết định có độ tin cậy thấp, trường hợp các ASR bất
đồng, trường hợp không có tiếng nói và một mẫu phân tầng từ mọi tác vụ. Khi báo
cáo, hãy cung cấp cả điểm tự động lẫn điểm sau hiệu chỉnh thủ công và giữ lại tệp
ghi các hiệu chỉnh.

## Kết quả tham chiếu GPT-Realtime đã hoàn tất

Chúng tôi đã chạy **GPT-Realtime** trên toàn bộ 400 mẫu event và 200 clean
control. Việc chấm semantic dùng prompt tiếng Việt **GPT-4.1-mini** có xét vai
trò hội thoại. Pilot có 29 hiệu chỉnh từ người Việt bản ngữ; expansion hiện chỉ
được chấm tự động và cần cùng quy trình kiểm tra thủ công trước khi công bố như
một kết quả leaderboard.

| Tác vụ | Pilot (20) | Expansion (30) | Toàn bộ (50) |
|---|---:|---:|---:|
| Backchannel | 100% | 100% | 100% |
| Xử lý khoảng dừng | 40% | 36,7% | 38% |
| Chuyển lượt mượt mà | 85% | 73,3% | 78% |
| Người dùng ngắt lời — bản chuẩn | 40% | 80% | 64% |
| Tiếng nói nền | 30% | 10% | 18% |
| Người dùng nói với người khác | 30% | 26,7% | 28% |
| Backchannel của người dùng | 100% | 96,7% | 98% |
| Người dùng ngắt lời — biến thể ghép cặp | 80% | 96,7% | 90% |
| **Tổng** | **63,1% (101/160)** | **65,0% (156/240)** | **64,2% (257/400)** |

Các metric về timing/chuyển lượt, không phụ thuộc ngôn ngữ và được xuất theo cấu
trúc tương thích với English FDB, gồm:

| Metric FDB gốc | Kết quả GPT-Realtime (50 mẫu) |
|---|---:|
| Tỷ lệ giành lượt trong khoảng dừng (càng thấp càng tốt) | 48% (24/50) |
| Tỷ lệ chờ đúng trong khoảng dừng | 52% (26/50) |
| Tỷ lệ giành lượt đúng sau khi người dùng nói xong | 98% (49/50) |
| Độ trễ chuyển lượt mượt mà, tính trên các mẫu có phản hồi | 1,000 giây |
| Tỷ lệ phản hồi sau khi bị ngắt lời | 76% (38/50) |
| Độ trễ phản hồi sau khi bị ngắt lời, tính trên các mẫu có phản hồi | 0,662 giây |

Điểm ngữ nghĩa và các metric timing trả lời những câu hỏi khác nhau. Ví
dụ, hệ thống có thể chờ đúng qua một khoảng dừng nhưng sau đó bỏ sót phần còn lại
của yêu cầu tiếng Việt. Metric timing ghi nhận việc chờ là đúng, trong khi bộ
chấm ngữ nghĩa đánh giá đầy đủ hành vi này là không đạt.

Điểm yếu rõ nhất trong pilot là xử lý lời nói không hướng đến hệ thống: cả tiếng
nói nền và người dùng nói với người khác chỉ đạt 30%. Hệ thống xử lý backchannel
tốt, nhưng các tác vụ khoảng dừng và thay đổi yêu cầu khi ngắt lời cho thấy lỗi
phản hồi quá sớm, bỏ lỡ ngắt lời và không tiếp nhận đầy đủ nội dung mới.

## Nội dung cần báo cáo cho một submission mới

Tối thiểu cần ghi lại:

- revision hoặc commit SHA chính xác của repository Vi-FDB;
- tập con được đánh giá (`pilot_160`, `expansion_240` hoặc cả hai);
- phiên bản mô hình/dịch vụ, giọng nói, cấu hình VAD/phát hiện lượt và ngày chạy;
- mức concurrency và việc các cặp event/clean có dùng session mới, tách biệt hay
  không;
- số mẫu hoàn tất và các mẫu lỗi/chạy lại;
- backend và phiên bản ASR;
- phiên bản mô hình/prompt dùng để chấm tự động;
- chính sách lấy mẫu kiểm tra thủ công và toàn bộ hiệu chỉnh;
- số mẫu đạt theo từng tác vụ, tỷ lệ đạt tổng thể và các metric timing phù hợp.

Khi so sánh submission, hãy báo cáo cả kết quả toàn bộ và kết quả trên cùng một
split có tên. Pilot đã được hiệu chỉnh thủ công, còn expansion hiện chưa có bước đó.

## Artifact tham chiếu trong repository bộ dữ liệu

Repository Hugging Face bao gồm:

- `evaluation/upstream_metrics.json`: metric timing tương thích với English FDB.
- `evaluation/semantic_parity.json`: số liệu semantic của pilot, expansion và toàn bộ.
- `evaluation/interruption_relevance_summary.json`: kết quả đánh giá mức độ liên
  quan sau ngắt lời đã bản địa hóa cho tiếng Việt.
- `data/pilot_160/vibe_check.html`: trang duyệt tập pilot.
- `data/*/manifest.json`: danh sách mẫu chuẩn của từng tập.

Để xem lý do thiết kế, các lỗi đã biết và giới hạn của bản phát hành, đọc
`docs/vi-fdb-v1-status-and-findings.md` trong repository của harness.
