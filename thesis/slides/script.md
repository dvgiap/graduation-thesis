# Script thuyết trình bảo vệ khóa luận
**Chủ đề:** Correlation-Aware Reward Exploration (CARE) cho phần thưởng nội tại
**Tác giả:** Đoàn Văn Giáp — UET, VNU
**Thời lượng dự kiến:** 13–14 phút trình bày + Q&A
**Tổng số slide:** 14 chính + 3 backup

> **Cách dùng:** Mỗi mục có (i) lời nói gợi ý — đọc tự nhiên, không học thuộc, (ii) ý chính cần truyền, (iii) thời gian ước lượng. Phần in nghiêng là gợi ý cử chỉ/nhấn nhá. Các con số quan trọng đã được kiểm tra khớp với báo cáo và phụ lục.

---

## Slide 1 — Trang bìa  *(~30s)*

> *Bước ra, cúi chào, đợi 1 nhịp.*

"Kính thưa Hội đồng, em xin tự giới thiệu, em là Đoàn Văn Giáp, sinh viên Khoa Công nghệ Thông tin, Trường Đại học Công nghệ. Hôm nay em xin được trình bày khóa luận tốt nghiệp với chủ đề *Nghiên cứu Động lực Nội tại trong Học Tăng cường*, mà cụ thể là phương pháp **Correlation-Aware Reward Exploration**, viết tắt là **CARE**, do TS. Nguyễn Thị Thủy hướng dẫn. Em xin được bắt đầu trình bày."

**Ý chính:** giới thiệu ngắn gọn, không kể lể. Đợi ánh nhìn của hội đồng rồi sang slide.

---

## Slide 2 — Bối cảnh: bài toán phần thưởng thưa  *(~60s)*

> *Đứng nghiêng người sang phải, tay chỉ vào cột bên trái.*

"Học tăng cường đã đạt nhiều thành tựu nổi bật ở các bài toán có **phần thưởng dày** — Atari, AlphaGo, robot điều khiển — vì tác tử nhận phản hồi từ môi trường rất thường xuyên, nên dễ học.

Tuy nhiên, **rất nhiều bài toán thực tế lại có phần thưởng cực thưa**: tín hiệu chỉ xuất hiện sau cả một chuỗi hành động dài. Khám phá ngẫu nhiên gần như không bao giờ chạm tới đích. Ví dụ điển hình là điều hướng phòng nhiều tầng, hay mở khóa rồi nhặt đồ.

Hướng tiếp cận phổ biến để xử lý trường hợp này là **bổ sung một phần thưởng nội tại** — driven bởi tò mò hay độ mới — vào phần thưởng ngoại sinh. Công thức điển hình là `r-bar = R_E + β · R_I`, trong đó **β là một hệ số cố định**, do người làm thí nghiệm chọn bằng tay. Các module nội tại đại diện gồm Count-based, ICM, RND, RIDE, RE3."

**Ý chính:** dẫn dắt từ sparse reward → intrinsic motivation → hệ số β cố định. Đây là setup cho slide 3.

---

## Slide 3 — Hạn chế của β cố định  *(~60s)*

> *Đứng giữa, hai tay cùng diễn tả "khi này thì... khi kia thì...".*

"Vấn đề cốt lõi mà khóa luận này nhắm tới là: một giá trị β **duy nhất** không thể tốt cho **mọi trạng thái**.

Trực giác đơn giản: ở **đầu episode**, khi tác tử chưa biết đường, **cần khám phá mạnh**, nên β nên lớn. Nhưng khi tác tử đã thấy đích, **cần khai thác** — phải dập tò mò xuống, nếu không nó sẽ bị phân tâm. Một β cố định không phân biệt được hai loại trạng thái này.

Có ba bằng chứng kinh nghiệm rất rõ:
- **Thứ nhất**, β tối ưu khác nhau giữa các môi trường, chênh nhau tới 10 lần.
- **Thứ hai**, β tối ưu **thay đổi theo giai đoạn** huấn luyện.
- **Thứ ba**, β **rất nhạy**. Trong thí nghiệm của em, giá trị β bằng 0.05 đã đủ phá nát hầu hết các môi trường.

Vậy nên cần một cơ chế **tự điều chỉnh β theo từng trạng thái**, học trực tuyến."

**Ý chính:** dẫn motivation chặt → câu "cần một cơ chế..." nói chậm lại, đó là cầu sang slide 4.

---

## Slide 4 — Câu hỏi nghiên cứu & Đóng góp  *(~75s)*

> *Đọc khung block chậm, nhấn từ "phụ thuộc trạng thái".*

"Câu hỏi nghiên cứu của khóa luận, em trích nguyên văn: *Có thể học một hàm β(s) phụ thuộc trạng thái, từ chính trải nghiệm tác tử, để cải thiện cách các module phần thưởng nội tại đóng góp vào việc học, trong môi trường phần thưởng thưa hay không?*

Để trả lời câu hỏi này, khóa luận có **ba đóng góp** chính:

**Một**, đề xuất **CARE** — một lớp scaling thích ứng, học β-ψ(s) bằng một **mục tiêu tương quan** với GAE advantage ngoại sinh. Cụm từ này nghe phức tạp, em sẽ giải thích chi tiết ở vài slide tới.

**Hai**, đặc biệt nhấn mạnh **tính tổng quát theo module**: cùng một cơ chế CARE, không đổi gì, áp dụng cho **ba họ nội tại khác biệt** — count-based, ICM, và RIDE.

**Ba**, **đánh giá thực nghiệm quy mô lớn — 570 lần chạy** trên 6 môi trường MiniGrid. Kết quả là CARE khớp hoặc vượt giá trị β cố định tốt nhất ở **7 trên 18 cặp (môi trường, module)** — và quan trọng hơn — **mà không cần grid-search β** thủ công."

**Ý chính:** "570 runs" và "7/18" là hai con số nhấn mạnh. Hội đồng RL sẽ ấn tượng với scale này.

---

## Slide 5 — Nền tảng: PPO + 3 module curiosity  *(~40s)*

> *Lướt nhanh — slide kiến thức nền, không sa đà.*

"Phần nền tảng em đi nhanh. **PPO** là backbone — actor-critic, clipped surrogate, GAE λ=0.95, 80 epoch / 4000 bước — **dùng chung cho mọi cấu hình** để so sánh công bằng.

**Ba module nội tại**: **Count-based** thưởng trạng thái hiếm; **ICM** dùng lỗi forward-model; **RIDE** là tín hiệu impact-driven có giảm dần trong episode.

Trước khi vào CARE, mọi module đều **z-score và clip dương** — loại bỏ ảnh hưởng thang đo gốc."

**Ý chính:** Nói chậm 3 dòng nội tại; câu chuẩn hoá z-score là cầu sang slide CARE.

---

## Slide 6 — CARE: ý tưởng cốt lõi  *(~70s)*

> *Tay phải chỉ vào hộp công thức.*

"Đây là ý tưởng then chốt của CARE: thay β cố định bằng một **mạng nhỏ β-ψ(s)** — em gọi là **Beta Network**.

Công thức shaped reward trở thành: `r-bar = R_E + β-ψ(s) · I+`. β-ψ(s) bị **clip vào khoảng `[10⁻⁴, 5×10⁻²]`** — em sẽ giải thích cận trên 5×10⁻² ở slide kết quả.

Beta Network gồm encoder 2 lớp 256 chiều, đầu ra 128, sinh ra `log β-ψ`. Lý do dùng log: vì β nằm trong khoảng spanning hai bậc giá trị, học trong log-space ổn định hơn nhiều.

Em đặt **anchor `β₀ = √(β_min · β_max) ≈ 2.24×10⁻³`** — đây là **trung bình hình học**, tương ứng với **trung điểm log-space**. Anchor này không phải tuỳ ý: nó là điểm khởi đầu đối xứng tự nhiên nhất, không thiên vị về aggressive exploration hay zero-intrinsic.

**Trực giác của Beta Network**: khuếch đại tò mò ở những trạng thái mà chính sách đang **vượt kỳ vọng giá trị**, và dập tò mò ở các trạng thái còn lại."

**Ý chính:** anchor β₀ là geometric mean — đây là chi tiết thiết kế hay, hội đồng có thể hỏi. Câu cuối là dẫn vào meta-loss ở slide 7.

---

## Slide 7 — CARE: mục tiêu huấn luyện β-ψ  *(~75s)*

> *Để công thức trên màn hình một nhịp dài hơn — đây là phần kỹ thuật cốt lõi.*

"Để học β-ψ, em định nghĩa **meta-loss** sau:

`L_β = -E[ Î_z · Â^E_z ] + λ_reg · (log β-ψ - log β₀)²`

Trong đó:
- `Î = β-ψ(s) · I+` là phần thưởng nội tại đã có trọng số.
- `Â^E` là GAE advantage ngoại sinh **thô**, chưa chuẩn hoá. Lý do dùng "thô" em sẽ nói ngay sau đây.
- Cả hai đại lượng được **z-score trong minibatch**, nên thành phần đầu tiên chính là **tương quan Pearson trong batch**.
- `λ_reg = 10⁻³` là regularizer kéo log β-ψ về log β₀.

Phần này nhìn phức tạp nhưng **diễn giải gradient lại rất rõ**:
- Nếu trong batch, ở những trạng thái có intrinsic cao mà advantage cũng dương — **covariance dương** — gradient sẽ **tăng β**.
- Ngược lại, intrinsic cao mà advantage âm — **covariance âm** — gradient sẽ **giảm β**.
- Còn nếu covariance gần 0 — như trong **cold-start sparse reward** — regularizer chiếm ưu thế, β neo về β₀.

Tức là CARE **không huỷ hoại huấn luyện** khi không có tín hiệu — nó chỉ rơi về behavior của một β cố định bằng β₀."

**Ý chính:** ba bullet về gradient là điểm chốt. Câu cuối "không huỷ hoại huấn luyện" là defense mechanism cho hội đồng.

---

## Slide 8 — Tại sao dùng advantage thay return  *(~60s)*

> *Slide này là điểm khác biệt với paper gốc — nhấn mạnh "lệch một bước".*

"Một điểm em muốn nhấn mạnh là **lựa chọn tín hiệu giám sát** cho Beta Network. Lựa chọn tự nhiên có thể là **discounted return G^E**. CARE dùng **GAE advantage Â^E** thay vì return.

Vì sao? Vì G^E có một **vấn đề structural** trong môi trường sparse: variance của nó bị chi phối bởi **vị trí trong episode**, chứ không phải bởi trạng thái đó tốt hay xấu. Trạng thái đầu episode "thấy" gần như toàn bộ chuỗi reward; trạng thái cuối episode "thấy" gần 0. Như vậy tín hiệu correlation sẽ bị nhiễu bởi một biến confound.

Â^E = G^E - V(s) **trừ baseline value function**. Việc trừ baseline này **loại bỏ confound vị trí episode**, để tín hiệu còn lại đúng nghĩa là "vượt kỳ vọng baseline bao nhiêu" — đây mới đúng là credit per-state.

Điều này khớp đúng với trực giác đã nói: khuếch đại nội tại ở những state mà chính sách đang **vượt expectation**.

Cài đặt quan trọng: Beta Network nhận **advantage thô, không normalize**, để giữ thông tin độ lớn. PPO policy update vẫn dùng advantage đã normalize như chuẩn mực."

**Ý chính:** lý do dùng advantage thay vì return là điểm thiết kế đặc trưng. Hội đồng có thể hỏi "tại sao không dùng return cho đơn giản" — câu trả lời ở đây.

---

## Slide 9 — Thiết kế thực nghiệm  *(~40s)*

> *Lướt nhanh các bullet, nhấn các con số: 19, 5, 6, 570, 10⁶.*

"Phần thực nghiệm: **6 môi trường MiniGrid** sparse-reward — cover các dạng khám phá khác nhau từ DoorKey tới UnlockPickup.

Mỗi env có **19 cấu hình** (1 PPO baseline + 3×5 β cố định + 3 CARE), × 5 seeds × 6 envs = **570 lần chạy**, mỗi seed 1 triệu bước trên T4.

**Bốn chỉ số**: episode return, sample efficiency (đạt 0.7×asymptotic), stability (std cross-seed), và phân tích β học được."

**Ý chính:** scale 570 runs là điểm cộng. Nhấn "không cần grid-search" sau khi nói xong baseline để tự nhiên.

---

## Slide 10 — Kết quả 1: CARE-COUNT & CARE-RIDE thắng/hoà  *(~75s)*

> *Đưa tay chỉ vào ảnh aggregate_summary, giọng phấn khởi hơn ở slide này.*

"Đến phần kết quả. Hình bên trái là tổng hợp 18 cặp (môi trường, module) — sample efficiency và normalized final reward.

Tổng quan:
- CARE **khớp hoặc vượt β cố định tốt nhất**: **7/18** — gồm 3 vượt thực sự và 4 hoà trong ±0.005 reward.
- Kém: **11/18**, tập trung phần lớn ở module ICM và môi trường RedBlueDoors.

Hai kết quả nổi bật em muốn nhắc:

**Một**, **CARE-RIDE trên UnlockPickup**: 0.536 so với 0.369 — **gain +0.167**. Đây là môi trường có hai giai đoạn mở khoá rồi nhặt đồ, β cố định không xử lý nổi sự thay đổi nhu cầu khám phá giữa hai pha; CARE thì có.

**Hai**, **CARE-COUNT trên Empty-16x16** đạt ngưỡng 0.7 chỉ sau **66 nghìn bước**, so với 82k của β cố định tốt nhất — **tốc độ học cũng tốt hơn**.

Và variance của CARE-COUNT là thấp nhất trên mọi môi trường mà nó hội tụ — bằng chứng tính ổn định cross-seed."

**Ý chính:** Đọc rành mạch hai con số gain. Nhấn "không cần grid-search" thêm lần nữa.

---

## Slide 11 — Kết quả 2: β-ψ(s) thật sự phụ thuộc trạng thái  *(~70s)*

> *Chỉ vào hai hình lần lượt, không trộn lẫn.*

"Một câu hỏi tự nhiên: **Beta Network có thật sự học được hàm phụ thuộc trạng thái không, hay nó collapse về một hằng số?** Hai hình ở đây cho thấy câu trả lời.

Bên trái là **dynamics của CARE-COUNT** qua thời gian. Bên phải là **histogram cuối training của CARE-RIDE**, với các vạch đứng đánh dấu β₀ và 5 giá trị fixed trong sweep.

Có ba chế độ rõ ràng:
- **Một**, ở môi trường extrinsic xuất hiện sớm — DoorKey, Empty — β-ψ **tách khỏi β₀ trong khoảng 10 update đầu** và **giảm về phía β_min**. Tác tử nhận ra: đã có signal ngoại sinh, không cần khám phá nhiều nữa.
- **Hai**, ở môi trường extrinsic xuất hiện muộn — KeyCorridor, RedBlueDoors, UnlockPickup — β-ψ **neo tại β₀** trong giai đoạn cold-start, **chỉ tách khi advantage có magnitude**. Đây đúng là regime mà em chứng minh chính thức trong Corollary 3.1 ở chương 3.
- **Ba**, **phân phối cuối trải khoảng một bậc giá trị qua các state** — đây là bằng chứng định lượng rằng β-ψ học **không phải một hằng số**, mà là một hàm thật sự phụ thuộc trạng thái."

**Ý chính:** ba chế độ là phát hiện quan trọng. Có thể nhắc "Corollary 3.1 ở chương 3" để show rằng kết quả thực nghiệm khớp với lý thuyết.

---

## Slide 12 — Phân tích thất bại: vì sao CARE-ICM kém  *(~70s)*

> *Giọng nói trung tính, không né tránh điểm yếu — đây là điểm cộng cho academic honesty.*

"Em muốn dành một slide để **không né tránh điểm yếu**: CARE-ICM kém. Hình bên trái là performance profile của module ICM — đường CARE bị các giá trị β cố định nhỏ vượt qua trên hầu hết khoảng τ.

Trường hợp cực đoan là **KeyCorridor**: β cố định 0.001 đạt 0.457, còn CARE-ICM **bằng 0 trên cả 5 seed**.

Cơ chế thất bại em đã phân tích:
- Tín hiệu ICM dựa trên **lỗi forward-model**. Vùng có lỗi lớn **không nhất thiết** nằm trên đường tới đích — chỉ là nơi model dự đoán sai.
- Trong khi đó, advantage ngoại sinh có thể tình cờ dương ở những vùng này.
- Covariance lúc đó dương — nhưng là **dương nhầm**.
- Gradient meta-loss **khuếch đại β-ψ** ở vùng ngoài đường tới đích.
- Histogram CARE-ICM trên KeyCorridor là **bimodal, đuôi nặng gần β_max**.

**Bài học rút ra**: CARE **không phải là replacement** cho việc chọn module nội tại tốt. Nó là một bộ nhân theo state, mà chất lượng phụ thuộc vào **độ tương quan** giữa tín hiệu nội tại thô và đường tới extrinsic success. Khi module thô có alignment tốt — như count-based hay RIDE — CARE giúp; khi module thô lệch — như ICM trong gridworld — CARE có thể làm tệ hơn."

**Ý chính:** thừa nhận yếu điểm rõ ràng + giải thích cơ chế. Hội đồng đánh giá cao academic honesty kèm phân tích kỹ.

---

## Slide 13 — Kết luận & Giới hạn  *(~75s)*

> *Tóm tắt phẳng, không lặp lại nguyên các câu trước.*

"Tóm tắt **đóng góp đã được kiểm chứng**:
- Một **cơ chế scaling theo trạng thái** cho phần thưởng nội tại, học qua mục tiêu tương quan với GAE advantage.
- **Cùng cơ chế áp dụng cho 3 module**, không cần thiết kế lại — chứng minh tính module-agnosticism.
- Trên **count-based và RIDE**, CARE khớp hoặc vượt giá trị β cố định tốt nhất mà không cần grid-search trên môi trường mới.

**Giới hạn em xin được nêu thẳng thắn**:
- **Phụ thuộc vào chất lượng module nội tại**: CARE không cứu được ICM khi tín hiệu thô không khớp đường-tới-đích.
- **Cold-start trên môi trường cực thưa** như Empty-16x16: khi advantage gần 0 ở mọi state, gradient correlation gần như là noise, β-ψ neo về β₀ và CARE **thoái biến về behaviour của một β cố định**. Đây là failure mode đã được lường trước trong section VI của báo cáo.
- **Phạm vi MiniGrid**: action rời rạc, observation thấp chiều. Em chưa kiểm chứng trên continuous control hoặc môi trường có observation phức tạp."

**Ý chính:** thừa nhận giới hạn cụ thể, có dẫn chiếu (section VI, Corollary). Tránh "hy vọng tương lai sẽ tốt hơn" — quá generic.

---

## Slide 14 — Hướng phát triển & Q&A  *(~60s)*

> *Đứng giữa, mỉm cười, đợi vài giây sau "Xin cảm ơn".*

"**Hướng phát triển**:
- **Mở rộng phạm vi**: continuous control, multi-task RL, embodied navigation — nơi tín hiệu nội tại có cấu trúc khác gridworld.
- **Phân tích lý thuyết sâu hơn** cho mục tiêu tương quan — khi nào adaptive scaling chắc chắn có lợi.
- **Tín hiệu giám sát thay thế** cho Beta Network: bootstrapped novelty, trajectory-level success — để khắc phục regime cold-start cực thưa.
- **Lai ghép** CARE với meta-gradient hoặc diversity-based exploration.

Em xin kết thúc phần trình bày tại đây. **Xin chân thành cảm ơn Hội đồng** đã lắng nghe. Em sẵn sàng nhận các câu hỏi và góp ý từ Hội đồng."

> *Cúi nhẹ, không bước lùi vội. Đợi câu hỏi.*

---

# Checklist Q&A — chuẩn bị trước

## Câu hỏi rất có khả năng được hỏi

**Q1. Vì sao chọn `[10⁻⁴, 5×10⁻²]` làm cận của β-ψ?**
> Cận trên 5×10⁻² là **trực tiếp từ kết quả thực nghiệm**: trong sweep β cố định, giá trị 0.05 phá nát hầu hết môi trường — đây là evidence-based clipping. Cận dưới 10⁻⁴ thấp hơn một bậc giá trị so với giá trị nhỏ nhất trong sweep (5×10⁻⁴), cho phép CARE undershoot nếu cần.

**Q2. β-ψ có thể overfit/collapse không?**
> Có hai cơ chế chống collapse: (i) **log-space regularizer** kéo về β₀, (ii) **clipping cứng** trong khoảng `[β_min, β_max]`. Empirically, histogram cuối training trải ~1 bậc giá trị — không collapse về điểm.

**Q3. Vì sao không dùng intrinsic-only advantage thay vì extrinsic advantage?**
> Vì mục tiêu cuối cùng là maximize **extrinsic return**. Học β để correlation với intrinsic advantage là circular — sẽ amplify tò mò một cách tự xác nhận, không gắn với việc tác tử có giải được task hay không.

**Q4. So với meta-gradient method (như học β bằng meta-grad qua return), CARE rẻ hơn ở đâu?**
> Meta-gradient cần backprop qua **toàn bộ inner loop PPO** — chi phí RAM và compute lớn. CARE chỉ cần **một bước forward+backward trên Beta Network mỗi outer PPO iteration**, không backprop qua policy. Khác biệt một bậc giá trị về wallclock.

**Q5. CARE thoái biến về β₀ trong cold-start — vậy chọn β₀ có quan trọng không?**
> Cực kỳ. Em đặt β₀ = √(β_min · β_max), tức **trung điểm log-space của khoảng cho phép**, là điểm không thiên vị giữa "khám phá mạnh" và "tắt khám phá". Một β₀ thiên lệch sẽ làm CARE thoái biến về một fixed-β chệch.

## Câu hỏi có thể được hỏi

**Q6. Vì sao chỉ dùng MiniGrid, không thử Procgen / Atari?**
> MiniGrid cho phép kiểm soát chính xác **mức độ sparse** và **cấu trúc thưởng**, là test bed sạch nhất để kiểm chứng giả thuyết về adaptive scaling. Mở rộng sang env phức tạp hơn là future work.

**Q7. Z-score trong minibatch có ổn định khi batch nhỏ không?**
> Rollout buffer là 4000 bước — batch tương đối lớn so với mini-batch SGD thông thường. Em không quan sát instability từ z-score trong các log.

**Q8. CARE có làm tăng wallclock so với fixed-β không?**
> Có, nhưng nhỏ. T4 GPU: ~2 giờ cho no-intrinsic, ~3 giờ cho fixed-β, ~4 giờ cho CARE — tăng chủ yếu do auxiliary networks (encoder ICM/RIDE + Beta Network).

**Q9. Có thử Polyak target cho β-ψ không?**
> Có thử qua trong giai đoạn early development, không cải thiện và làm code phức tạp hơn. Quyết định cuối là dùng β-ψ live theo Algorithm 1, không Polyak target.

**Q10. Vì sao CARE-ICM kém ở KeyCorridor nhưng thắng ở LavaCrossing?**
> LavaCrossing có ít vùng "lỗi forward-model lớn nhưng không trên đường tới đích" — khu vực ngoài đường là lava chết. KeyCorridor có nhiều phòng rỗng nơi ICM error cao dai dẳng. Khác biệt là **topology của extrinsic-success path**.

## Câu hỏi khó

**Q11. Mục tiêu tương quan có guarantee gì về convergence của policy không?**
> Không có guarantee chặt. Em chứng minh được **direction of gradient** (Proposition 3.3) và **degenerate regime** (Corollary 3.1), nhưng convergence của joint optimization policy + β-ψ là open problem — đây cũng là một future direction nêu ở slide 14.

**Q12. Nếu intrinsic reward đã được tune kỹ thì CARE còn cần không?**
> Trên một môi trường cụ thể, một β tune kỹ có thể tốt bằng CARE. Nhưng CARE **transferable**: từ DoorKey sang KeyCorridor không cần tune lại; β cố định tốt cho DoorKey có thể chệch 10 lần cho KeyCorridor.

**Q13. Tại sao 5 seeds là đủ?**
> Là chuẩn de-facto cho RL benchmark (Andrychowicz et al. 2021, Engstrom et al. 2020). Em báo cáo std cross-seed; với những kết quả em claim (gain ≥ 0.15 ở UnlockPickup), khoảng tin cậy bootstrap không phủ 0.

---

# Hướng dẫn compile slide

```bash
# Từ thư mục thesis/slides/
pdflatex slides.tex
pdflatex slides.tex   # chạy lần 2 để cập nhật cross-reference

# Hoặc dùng latexmk (nếu có)
latexmk -pdf slides.tex

# Hoặc Overleaf: upload thư mục thesis/ rồi chọn slides/slides.tex làm main file
```

**Lưu ý:**
- Font tiếng Việt qua package `vietnam`. Nếu compile ở Linux/Mac bị lỗi font, thay `\usepackage[utf8]{vietnam}` bằng `\usepackage[vietnamese]{babel}` + `\usepackage[utf8]{inputenc}`.
- Đường dẫn `\graphicspath{{../figures/}}` đã trỏ đúng tới `thesis/figures/` — không cần copy ảnh.
- Theme `Madrid + seahorse` — nếu thầy/cô muốn theme khác (vd `metropolis` hiện đại), đổi 2 dòng `\usetheme` và `\usecolortheme`.

# Tips trình bày

1. **Tập luyện với đồng hồ** — đặt mục tiêu 12 phút để có buffer 2–3 phút phòng khi hội đồng cắt ngang.
2. **Slide 7 (meta-loss) là điểm cao kỹ thuật nhất** — luyện kỹ phần này, đừng đọc công thức như đọc từ điển.
3. **Slide 12 (failure mode CARE-ICM) thường được hỏi sâu** — chuẩn bị kỹ phần "cơ chế thất bại" để trả lời ngay nếu được follow-up.
4. **Không trốn câu hỏi về giới hạn** — nói thẳng "đây là failure mode đã biết, được mô tả ở section VI báo cáo" là đủ.
5. **Nếu mic / projector trục trặc** — backup PDF trong USB và một bản trên drive.
