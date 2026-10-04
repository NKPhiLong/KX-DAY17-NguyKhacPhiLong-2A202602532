# Phân tích kết quả — Day 17: Memory Systems for AI Agent

Tài liệu này giải thích các con số do `python src/benchmark.py` sinh ra. Mỗi luận điểm đi theo ba bước: **số liệu → cơ chế trong `src/` tạo ra số liệu đó → giới hạn đi kèm**.

- Số liệu offline (số liệu chính): mục 2. Số liệu live với gpt-4o-mini: mục 5.
- Test: `pytest src/test_agents.py -v`, 8/8 pass (mục 6).

## 1. Cách chạy lại

```bash
source .venv/bin/activate
rm -rf state
python src/benchmark.py
pytest src/test_agents.py -v
```

Mọi số liệu dưới đây là của **chế độ offline tất định**: không cần API key, hai lần chạy liên tiếp trên state sạch ra cùng một output (đã kiểm tra bằng `diff`, và có test `test_benchmark_is_deterministic`).

Quy ước biến môi trường (`src/config.py`, đều tùy chọn, có thể đặt trong `.env`):

| Biến | Ý nghĩa | Mặc định |
|---|---|---|
| `LLM_MODE` | `live` để gọi model thật; mọi giá trị khác = offline | `offline` |
| `LLM_PROVIDER`, `LLM_MODEL`, `LLM_TEMPERATURE` | model chính: `openai`, `custom`, `gemini`, `anthropic`, `ollama`, `openrouter` (chấp nhận alias như `anthorpic`, `claude`) | `openai`, `gpt-4o-mini`, `0.0` |
| `JUDGE_PROVIDER`, `JUDGE_MODEL`, `JUDGE_TEMPERATURE` | model judge | giống `LLM_*` |
| `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY` | API key theo provider | — |
| `CUSTOM_BASE_URL`, `CUSTOM_API_KEY`, `OLLAMA_BASE_URL` | endpoint cho `custom` / `ollama` | — |
| `COMPACT_THRESHOLD_TOKENS`, `COMPACT_KEEP_MESSAGES` | ngưỡng compact / số message giữ nguyên văn | `600`, `4` |
| `STATE_DIR` | nơi ghi `User.md` | `state/` |

Chế độ live phải bật rõ bằng `LLM_MODE=live`. Lý do: nếu chỉ dựa vào việc có key, một `OPENAI_API_KEY` có sẵn trong shell sẽ âm thầm biến benchmark thành live và kết quả không còn lặp lại được.

## 2. Kết quả benchmark

Cấu hình: ngưỡng compact 600 token, giữ 4 message, `estimate_tokens = max(1, len // 4)`.

**Standard Benchmark** — `conversations.json`: 10 hội thoại, 101 lượt, 14 câu hỏi recall.

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|--:|--:|--:|--:|--:|--:|
| Baseline | 638 | 14,203 | 0% | 0.20 | 0 | 0 |
| Advanced | 588 | 24,494 | 100% | 1.00 | 407 | 0 |

**Long-Context Stress Benchmark** — `advanced_long_context.json`: 1 hội thoại, 16 lượt, 3 câu hỏi recall.

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|--:|--:|--:|--:|--:|--:|
| Baseline | 131 | 21,748 | 0% | 0.20 | 0 | 0 |
| Advanced | 132 | 9,559 | 100% | 1.00 | 264 | 9 |

Định nghĩa các cột (`src/benchmark.py`):

- **Agent tokens only** — tổng `estimate_tokens(câu trả lời)` của agent, cộng trên thread hội thoại và thread recall.
- **Prompt tokens processed** — tổng ngữ cảnh agent phải đọc ở **từng lượt**:
  - Baseline: system prompt + toàn bộ lịch sử thread.
  - Advanced: system prompt + `User.md` + summary + các message gần nhất.
- **Cross-session recall** — trung bình `recall_points` (1 / 0.5 / 0), hỏi ở thread mới `<conv_id>::recall`. Cả hai agent dùng cùng cách đặt tên thread này.
- **Response quality** — `heuristic_quality`, dùng cùng một công thức cho cả hai agent:
  - 0.6 × tỉ lệ fact có trong câu trả lời,
  - \+ 0.2 nếu agent trả lời thay vì nói "chưa có thông tin",
  - \+ 0.2 nếu câu trả lời ngắn gọn (≤ 80 token).
- **Memory growth** — kích thước `User.md` sau khi chạy trừ trước khi chạy. Mỗi bộ benchmark có thư mục state riêng (`state/benchmark/<suite>/`), bị xóa trước mỗi lần chạy.
- **Compactions** — tổng `compaction_count()` trên các thread.

## 3. Năm mắt xích của câu chuyện

### 3.1 Baseline không nhớ dài hạn

- **Số liệu:** recall của Baseline là **0%** ở cả hai bảng. Memory growth bằng **0**.
- **Cơ chế:** `BaselineAgent.sessions` được khóa theo `thread_id`, và `user_id` không được dùng.
  - Fact chỉ nằm trong `SessionState.facts` của thread hiện tại, trong RAM.
  - Thread `conv-01::recall` bắt đầu rỗng, nên câu trả lời là "mình chưa có thông tin này".
  - Trong cùng một thread, Baseline vẫn nhớ. `test_cross_session_recall` kiểm tra cả hai vế.
- **Giới hạn:** điểm 0% này là kết quả mong muốn của thiết kế, không phải lỗi. Khi thử cài lại lỗi "lưu session theo `user_id`", 2 test fail (xem mục 6).

### 3.2 `User.md` làm recall tăng

- **Số liệu:** recall của Advanced là **100%** ở cả hai bảng (14/14 và 3/3 câu đạt điểm tối đa). Memory growth dương: **407 B** và **264 B**.
- **Cơ chế:** đường đi của một fact gồm bốn bước:
  1. `extract_profile_updates()` trích fact.
  2. `apply_profile_updates()` ghi fact vào `state/profiles/<user>/User.md`.
  3. Ở thread mới, `_offline_response()` đọc `profile_store.facts(user_id)`.
  4. `compose_offline_reply()` ghép câu trả lời. Nếu style người dùng chọn có "bullet", câu trả lời được trình bày bằng bullet, nên câu trả lời cho user stress chứa đúng chuỗi "3 bullet".
- **Kiểm chứng bằng cách tắt lớp:** giữ compact nhưng tắt ghi `User.md` (monkeypatch). Recall của Advanced rơi về **0%** ở cả hai bộ.
  - Kết luận: toàn bộ recall đến từ lớp persistent, compact không đóng góp gì vào recall.
  - Đây là lý do `User.md` và compact là hai đường độc lập: recall không phụ thuộc vào việc summary có giữ được tên hay không.
- **Giới hạn:** điểm 100% có được vì câu trả lời offline được ghép từ chính các fact, và bộ chấm chỉ kiểm tra chuỗi con. Với model thật, câu trả lời có thể diễn đạt khác đi, và recall theo chuỗi con sẽ thấp hơn dù agent vẫn nhớ đúng.

### 3.3 Vì sao Advanced tốn hơn ở hội thoại ngắn

- **Số liệu (bảng Standard):**
  - Prompt tokens của Advanced là **24.494**, của Baseline là **14.203**, tức Advanced tốn hơn **72%**.
  - Agent tokens gần bằng nhau: **588** so với **638**.
  - Compactions là **0**.
- **Cơ chế:** mỗi lượt của Advanced mang thêm hai thứ mà Baseline không có. Mình tách hai phần này bằng phép tắt từng lớp:

  | Biến thể (Standard) | Prompt tokens |
  |---|--:|
  | Baseline | 14,203 |
  | Advanced, tắt `User.md` | 15,701 (cao hơn khoảng 1.500, do system prompt dài hơn) |
  | Advanced đầy đủ | 24,494 (thêm khoảng 8.800, do mang `User.md` vào từng lượt) |

  - `User.md` của `dungct` lớn dần lên khoảng 100 token. File này được đưa vào prompt ở cả 115 lượt (101 lượt hội thoại và 14 câu hỏi recall).
  - Mỗi thread standard chỉ có khoảng 280 token, dưới ngưỡng 600. Compact không chạy, nên không có gì bù lại phần chi phí cố định này.
- **Vì sao Agent tokens only không phân biệt được hai agent:** cột này chỉ đo phần agent tự sinh ra.
  - Ở chế độ offline, đa số lượt nhận câu trả lời template "Đã ghi nhận.".
  - Ở câu hỏi recall, Baseline nói "mình chưa có thông tin này", câu này còn dài hơn chính các fact. Vì vậy Baseline có số cao hơn một chút.
  - Chi phí của memory nằm ở cột **Prompt tokens processed**, không nằm ở cột này. Với model thật, câu trả lời của Advanced sẽ dài hơn vì có nội dung thật, và chênh lệch ở cột này sẽ lộ ra.
- **Giới hạn:** ghi `User.md` còn tốn I/O ở các lượt có fact mới. Benchmark không đo chi phí này.

### 3.4 Vì sao compact thắng ở hội thoại dài, và vì sao nó tối ưu **Prompt tokens processed**

- **Số liệu (bảng stress):**
  - Prompt tokens của Advanced là **9.559**, của Baseline là **21.748**, tức Advanced dùng ít hơn **56%**.
  - Có **9** lần nén.
  - Agent tokens gần như không đổi: **132** so với **131**.

  Prompt mỗi lượt (thread hội thoại):

  | Lượt | 1 | 2 | 3 | 4 | 6 | 8 | 10 | 12 | 14 | 16 |
  |---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
  | Baseline | 225 | 379 | 531 | 690 | 1,009 | 1,302 | 1,540 | 1,838 | 2,113 | 2,391 |
  | Advanced | 278 | 432 | 584 | 468 | 551 | 604 | 699 | 570 | 690 | 654 |
  | Compactions (cộng dồn) | 0 | 0 | 0 | 1 | 2 | 4 | 5 | 7 | 8 | 9 |

- **Cơ chế:**
  - Prompt mỗi lượt của Baseline bằng toàn bộ lịch sử thread, nên tăng tuyến tính khoảng 150 token mỗi lượt. Tổng cộng qua các lượt vì vậy tăng theo bình phương số lượt.
  - `CompactMemoryManager.append()` kiểm tra ngưỡng sau mỗi message. Khi summary + message vượt 600 token:
    - các message cũ được gộp vào `summarize_messages()` (tối đa 6 dòng, mỗi dòng ≤ 140 ký tự),
    - chỉ 4 message gần nhất được giữ nguyên văn.
  - Nhờ vậy prompt của Advanced bị chặn quanh **450–700 token mỗi lượt**, bất kể thread dài bao nhiêu.
  - Ba lượt đầu, Advanced còn đắt hơn Baseline khoảng 53 token mỗi lượt, do system prompt và `User.md`. Từ lượt 4, khi compact bắt đầu chạy, Advanced rẻ hơn.
- **Compact tối ưu cột nào:** compact chỉ thay đổi thứ agent phải **đọc**, không thay đổi thứ agent **viết**.
  - Vì vậy Agent tokens only đứng yên (131 so với 132), còn Prompt tokens processed giảm 56%.
  - Nói "compact giúp tiết kiệm token" là chưa đủ chính xác. Đúng hơn là compact tiết kiệm **token ngữ cảnh**.
- **Kiểm chứng bằng cách tắt compact** (`COMPACT_THRESHOLD_TOKENS=100000000`, không sửa code): prompt tokens của Advanced ở bảng stress tăng lên **22.990**, cao hơn Baseline 6%, và Compactions bằng 0.
  - Kết luận: toàn bộ lợi thế 56% là do compact tạo ra.
  - Khi không có compact, Advanced chỉ còn là "Baseline cộng thêm `User.md`".
- **Giới hạn:**
  - **Nén gần như mỗi lượt.** Mỗi lượt stress khoảng 150–190 token, mà 4 message giữ lại cộng summary đã gần chạm 600. Vì vậy compact chạy 9 lần trong 16 lượt, tức gần như mỗi lượt từ lượt 4 trở đi. Với summary bằng LLM, mỗi lần nén là một lần gọi model, nên cần nâng ngưỡng hoặc nén theo khối lớn hơn.
  - **Summary heuristic làm mất ý tổng quát.** Summary chọn câu có từ khóa fact, nên ở cuối thread chỉ còn vài câu rời. Pattern "readiness, externality, uncertainty, efficiency" mà lượt 12 dặn phải giữ thì không còn. Recall không bị ảnh hưởng vì fact ổn định nằm trong `User.md`, nhưng câu hỏi về **nội dung** của thread dài sẽ không trả lời được.

### 3.5 Hệ thống mạnh hơn nhưng phức tạp hơn và cần guardrail

- **Số liệu:**
  - `User.md` của `dungct` sau từng hội thoại standard: `244 → 256 → 287 → 364 → 409 → 407 → 407 → 407 → 407 → 407` byte.
  - User stress: 264 byte, có **9** lần nén.
- **Cơ chế:**
  - File tăng nhanh trong 5 hội thoại đầu, khi fact mới xuất hiện. Sau đó file đứng ở khoảng 407 byte.
  - Lý do là các khóa đơn trị (tên, nơi ở, nghề…) được **thay thế** qua `edit_text()`, không ghi thêm dòng mới.
  - Từ hội thoại 5 sang 6, file còn nhỏ đi 2 byte, khi "backend engineer" được thay bằng "MLOps engineer".
- **Rủi ro quan sát được:**
  1. **Khóa dạng tập hợp chỉ tăng, không giảm.** `response_style` đã có 7 tag ("ngắn gọn, rõ ý, có ví dụ thực tế, có bullet, so sánh trade-off, có ví dụ số liệu, có cấu trúc"). `interests` có cả "benchmark memory", vốn chỉ là chủ đề đang đọc. Không có cơ chế nào xóa tag cũ, nên với người dùng thật dùng lâu, file sẽ phình và có thể chứa preference mâu thuẫn nhau.
  2. **Lưu sai fact khi bỏ guardrail.** Bỏ confidence threshold thì câu "đừng nói backend engineer nữa" bị ghi thành fact, và recall standard giảm còn 75% (mục 4).
  3. **Memory lâu dài ảnh hưởng mọi lượt sau.** Một fact sai trong `User.md` được đưa vào prompt ở mọi lượt và mọi thread sau đó, cho tới khi có correction. Baseline không có rủi ro này: nó quên ngay khi đổi thread.
  4. **Độ phức tạp tăng.** Advanced có thêm một file trên đĩa (cần làm sạch đường dẫn, xem `path_for()` và test đường dẫn có `../`), một bộ trích fact khoảng 150 dòng regex, và một bộ máy nén. Mỗi phần đều có thể sai theo cách riêng. Bốn test cài lỗi ở mục 6 tồn tại vì lý do này.

## 4. Bonus: Confidence threshold, Conflict handling, Entity extraction

Bonus được chọn theo điểm yếu thật của data: data cố tình chứa câu phủ định, câu đùa, câu nói về quá khứ, và các lần đổi nơi ở / đổi nghề.

**Cài đặt (`src/memory_store.py`):**

- **Entity extraction có cấu trúc.** Fact được tách thành 8 khóa: `name`, `location`, `profession`, `favorite_drink`, `favorite_food`, `pet`, `interests`, `response_style`.
  - Mỗi tin nhắn được tách thành câu, rồi thành mệnh đề, theo dấu phẩy và các từ "nhưng", "chứ", "dù".
  - Nơi ở chỉ nhận các tên thành phố trong danh sách, nên "sông Hương" hay "biển Mỹ Khê" không bị nhận là nơi ở.
- **Confidence threshold.** Mỗi `FactCandidate` có điểm confidence:
  - Điểm gốc theo độ mạnh của mẫu câu: "tên mình là" là 0.9, "ở X" là 0.75, "vẫn uống X" là 0.6.
  - \+0.1 nếu có dấu hiệu "hiện tại" hoặc correction ("giờ", "thực ra", "đính chính", "chuyển sang"…).
  - Phủ định ("không còn", "chứ không", "đừng") đặt điểm về **0.05**.
  - Quá khứ ("lúc đầu", "trước đó") −0.5. Nhiễu ("đùa", "chỉ là", "ra họp") −0.5. Câu giả định bắt đầu bằng "Nếu" −0.4.
  - Chỉ fact có điểm **≥ 0.6** mới được ghi.
  - Câu hỏi bị bỏ qua hoàn toàn, kể cả câu không có dấu "?" như "…đồ uống yêu thích của mình là gì."
- **Conflict handling.**
  - Khóa đơn trị: giá trị mới **thay thế** dòng cũ qua `upsert_fact()` → `edit_text()`. `User.md` luôn chỉ có một giá trị cho mỗi khóa.
  - Khóa tập hợp (`interests`, `response_style`): giá trị mới được gộp thêm, có loại trùng, và các từ đồng nghĩa được quy về một nhãn ("ví dụ thực tế" và "ví dụ thực chiến" là một tag).
  - Trong một tin nhắn, mệnh đề sau ghi đè mệnh đề trước. Ví dụ "Lúc đầu… ở Huế, nhưng thực ra… ở Đà Nẵng" cho kết quả Đà Nẵng.

**Bonus giải quyết vấn đề gì.** Các mẫu nhiễu và correction trong data bị loại hoặc được cập nhật đúng (in từ `extract_profile_candidates()`):

| Lượt | Ứng viên | Confidence | Kết quả |
|---|---|--:|---|
| conv-03#1 | location = Đà Nẵng ("không còn ở Đà Nẵng") | 0.05 | loại; Huế được ghi |
| conv-06#1, #8; conv-09#6 | profession = backend engineer (phủ định) | 0.05 | loại |
| stress#8 | location = Huế ("Lúc đầu mình nói hiện ở Huế") | 0.35 | loại; Đà Nẵng được ghi |
| stress#9 | profession = product manager ("đùa") | 0.40 | loại |
| stress#9 | "Hà Nội chỉ là nơi… họp" | — | không thành ứng viên |

**Bonus cải thiện recall và token cost bao nhiêu.** Đo bằng cách tắt từng phần bonus (monkeypatch), giữ nguyên mọi thứ khác:

| Biến thể | Recall Standard | Recall Stress | `User.md` (B) Std / Stress | Prompt tokens Std / Stress |
|---|--:|--:|--:|--:|
| **Đầy đủ** (threshold 0.6 + conflict handling) | **100%** | **100%** | 407 / 264 | 24,494 / 9,559 |
| Không có confidence threshold (ghi mọi ứng viên) | 75% | 100% | 409 / 264 | 24,530 / 9,559 |
| Không có conflict handling (ghi thêm dòng, đọc giá trị đầu tiên) | 64% | 67% | 796 / 404 | 30,830 / 9,936 |

- **Bỏ threshold.** Cuối bộ standard, `User.md` ghi `profession: backend engineer`, vì câu "chứ không còn là backend engineer nữa" (conv-09) được ghi như một fact.
  - Bộ stress vẫn đạt 100% chỉ vì các lượt sau tình cờ nhắc lại giá trị đúng.
  - Kết luận: threshold là thứ bảo vệ recall khi người dùng **không** nhắc lại fact đúng.
- **Bỏ conflict handling.** File chứa đồng thời `location: Đà Nẵng` và `location: Huế`, cùng `profession: backend engineer` và `profession: MLOps engineer`.
  - Recall giảm, vì bộ đọc lấy giá trị cũ.
  - File to **gấp đôi** (796 so với 407 byte).
  - Prompt standard tăng **26%**, vì file to hơn được đưa vào mọi lượt.
  - Kết luận: conflict handling vừa giữ đúng fact, vừa trực tiếp giảm token cost.

**Bonus tạo thêm rủi ro gì.**

- **Regex và từ khóa phụ thuộc ngôn ngữ, dễ vỡ.** Các ca dưới đây đã chạy thử với `extract_profile_candidates()`:

  | Câu | Kết quả | Loại lỗi |
  |---|---|---|
  | "Mình vừa chuyển từ Huế **ra** Đà Nẵng." | không trích được gì | bỏ sót fact đúng: mẫu câu chỉ hiểu "sang", không hiểu "ra" |
  | "Mình không ở Huế **mà** ở Đà Nẵng." | cả Huế lẫn Đà Nẵng đều 0.05, bị loại | bỏ sót fact đúng: "mà" không được dùng để tách mệnh đề, nên phủ định lan sang cả giá trị đúng |
  | "Mình **nghĩ là** mình đang ở Huế." | Huế, 0.85, được ghi | ghi nhầm: từ rào đón ("nghĩ là", "chắc là") không bị trừ điểm |
  | "Mình vẫn uống cà phê sữa đá." | 0.60, vừa đúng ngưỡng | gần như bỏ sót: chỉ cần mất chữ "vẫn" là fact không được ghi |

  Danh sách thành phố, từ vựng sở thích và từ khóa phủ định đều phải bảo trì thủ công. Mỗi cách diễn đạt mới là một chỗ có thể vỡ.
- **Dữ liệu cũ bị mất.** Thay thế giá trị nghĩa là **xóa lịch sử**. Agent không còn trả lời được "trước đây mình ở đâu?". Một hệ thống thật nên lưu lịch sử thay đổi ở chỗ khác (không đưa vào prompt) để còn kiểm tra lại và hoàn tác.
- **Ngưỡng được chọn theo data hiện có.** Các trọng số (0.05, −0.5, +0.1) được chỉnh trên bộ data này. Chúng chưa được kiểm chứng trên data khác, nên có nguy cơ overfit.

**Chưa làm:** memory decay. Rủi ro 1 ở mục 3.5 (tag style / sở thích chỉ tăng) là chỗ cần decay nhất: giảm ưu tiên các tag lâu không được nhắc lại, rồi xóa khi tụt dưới ngưỡng.

## 5. Chế độ live: chạy với OpenAI gpt-4o-mini

Lệnh chạy: `LLM_MODE=live python src/benchmark.py`, với key trong `.env`.

Kết quả live **không tất định**: model thật trả lời khác nhau giữa các lần chạy. Vì vậy số liệu chính của bài vẫn là số offline ở mục 2. Số live dùng để kiểm tra các kết luận có còn đúng với model thật hay không.

**Cách dựng agent live:**

- **Baseline:** `create_agent` + `InMemorySaver`. Checkpoint theo `thread_id`, nên vẫn chỉ nhớ trong thread.
- **Advanced:** `create_agent` với:
  - `InMemorySaver`,
  - tool `read_user_memory` / `save_user_fact` (ghi qua cùng hàm `apply_profile_updates()`),
  - `dynamic_prompt` chèn `User.md` vào prompt,
  - `SummarizationMiddleware(trigger=("tokens", 600), keep=("messages", 4))`.
- Bộ trích fact tất định vẫn chạy trước mỗi lượt, như một lớp an toàn.

**Kết quả live so với offline:**

| Bộ | Agent | Agent tokens | Prompt tokens | Recall | Quality | Memory growth | Compactions |
|---|---|--:|--:|--:|--:|--:|--:|
| Standard | Baseline | 6,078 | 43,309 | 11% | 0.45 | 0 | 0 |
| Standard | Advanced | 10,453 | 77,452 (+79%) | 100% | 0.99 | 1,039 | 28 |
| Stress | Baseline | 3,356 | 47,125 | 0% | 0.40 | 0 | 0 |
| Stress | Advanced | 3,299 | 19,569 (−58%) | 100% | 1.00 | 793 | 28 |

**Các kết luận vẫn đúng với model thật:**

- Advanced đạt recall 100% ở cả hai bộ.
- Ở hội thoại ngắn, Advanced tốn hơn **79%** prompt token. Offline là 72%.
- Ở hội thoại dài, compact giảm **58%** prompt token. Offline là 56%.

**Những điểm chỉ lộ ra khi chạy live:**

1. **Agent tokens only bắt đầu phân biệt được hai agent.** Ở bộ standard, Advanced sinh **10.453** token, còn Baseline sinh **6.078**.
   - Có profile trong prompt, model trả lời dài hơn và cá nhân hóa hơn.
   - Đây là khoản tốn thêm mà mục 3.3 dự đoán nhưng chế độ offline không đo được, vì offline chỉ trả lời bằng template.
   - Ở bộ stress, hai agent gần bằng nhau (3.299 so với 3.356). Compact vẫn chỉ làm giảm cột Prompt tokens.
2. **Recall 11% của Baseline là ảo.** Ba câu hỏi chứa sẵn đáp án:
   - conv-05: "Bạn biết **DũngCT** là ai…"
   - conv-06: "…mình còn ở **Huế** không?"
   - conv-09: "…mối quan tâm kỹ thuật" (có chuỗi "AI")

   Model thật lặp lại từ trong câu hỏi, nên bộ chấm chuỗi con vẫn cho nửa điểm. Đây là điểm yếu của cách chấm bằng chuỗi con, không phải bằng chứng Baseline nhớ được.
3. **`User.md` phình nhanh khi model tự ghi memory.** File standard tăng **1.039 B**, so với 407 B ở offline. File stress tăng **793 B**, so với 264 B.
   - Tool `save_user_fact` để model tự chọn ghi gì. Model đã ghi cả nội dung tạm thời vào `interests`, ví dụ "sửa dataset để các cuộc hội thoại tự nhiên hơn", "chi phí thật", "kế hoạch điện sạch".
   - `response_style` có tag trùng nghĩa: "ngắn gọn", "gọn hơn", "câu trả lời tự nhiên hơn".
   - Các fact đơn trị vẫn đúng: tên, nơi ở Huế / Đà Nẵng, nghề MLOps engineer. Lý do là conflict handling vẫn chặn ở `apply_profile_updates()`.
   - Đây là bằng chứng trực tiếp cho rủi ro 1 ở mục 3.5: guardrail cho các khóa dạng tập hợp (confidence threshold, memory decay, giới hạn số tag) cần thiết hơn khi để LLM tự ghi memory.
4. **Compactions = 28** ở cả hai bộ. Câu trả lời thật dài hơn template, nên thread standard cũng vượt ngưỡng 600.
   - Lưu ý: ở chế độ live, cột này đếm compact offline chạy song song với cùng ngưỡng. Nó **không** đếm số lần `SummarizationMiddleware` thật sự tóm tắt.

**Giới hạn còn lại:**

- Response quality vẫn là heuristic. `judge_model` đã có trong config nhưng chưa được nối vào chấm điểm.
- Prompt tokens ở chế độ live lấy từ `usage_metadata` của lần gọi model **cuối cùng** trong lượt. Nếu một lượt có gọi tool, các lần gọi model trước đó chưa được cộng vào, nên số live của Advanced có thể bị đếm thiếu.

## 6. Test và kiểm chứng test

`pytest src/test_agents.py -v`: **8/8 pass**, chạy trên `tmp_path` với ngưỡng 80 / giữ 2 message. Bốn test bắt buộc:

- `test_user_markdown_read_write_edit`
- `test_compact_trigger`
- `test_cross_session_recall` (khẳng định cả vế "Baseline **không** nhớ")
- `test_compact_reduces_prompt_load_on_long_thread`

Bốn test thêm: correction, nhiễu và câu hỏi, recall trên data stress thật, và benchmark tất định.

Để chắc test kiểm tra hành vi chứ không chỉ chạy qua, mình cài lại các lỗi trong mục "Lỗi thường gặp" vào một bản sao của `src/` rồi chạy test:

| Lỗi được cài vào | Test fail |
|---|--:|
| Baseline lưu session theo `user_id` | 2 |
| Advanced không ghi `User.md` | 3 |
| `append()` không kiểm tra ngưỡng, nên compact không chạy | 3 |
| Correction ghi thêm dòng mới thay vì `edit_text()` | 1 |

## 7. Đối chiếu Rubric

| Mốc | Yêu cầu | Trạng thái |
|---|---|---|
| 0–60 | Baseline chỉ nhớ trong thread; Advanced có `User.md`; compact chạy thật; dataset tiếng Việt | Đạt: recall Baseline 0%, `User.md` 407 / 264 B, 9 lần nén |
| 60–75 | Benchmark cùng input; test `User.md`, compact, cross-session; đủ 6 cột | Đạt: 8 test, 2 bảng × 6 cột |
| 75–90 | Có cả Standard và Stress; stress làm lộ chi phí của Baseline; giải thích vì sao compact không thắng ở hội thoại ngắn và vì sao nó tối ưu Prompt tokens | Đạt: mục 3.3, 3.4, kèm phép tắt compact |
| 90–100 | Bonus kèm "giải quyết gì, cải thiện gì, rủi ro gì" | Confidence threshold + conflict handling + entity extraction, đo bằng phép tắt bonus ở mục 4 |
