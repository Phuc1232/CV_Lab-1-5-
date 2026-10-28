# MULTI-AGENT ARCHITECTURE & WORKFLOW SPECIFICATION (AGENTS.MD)

Tài liệu quy định cấu trúc điều phối, vai trò trách nhiệm, mô hình ngôn ngữ và quy trình kiểm soát chất lượng của hệ sinh thái **Multi-Agent** áp dụng trong dự án.

---

## 1. Danh sách Sub-Agents & Cấu hình Mô hình (Model Roles)

Hệ thống vận hành theo cơ chế 4 Sub-Agents độc lập, tương hỗ và có ranh giới trách nhiệm nghiêm ngặt:

| Sub-Agent | Vai trò (Role) | Mô hình (Model Configuration) | Nhiệm vụ cốt lõi |
|:---:|:---|:---|:---|
| **Sub-1** | **Architect & Lead Orchestrator** | `gemini-3.8-flash-high-ultra-0.2` | Phân tích bài toán, phân rã kiến trúc, xây dựng đặc tả kỹ thuật, điều phối luồng làm việc, phản biện giải pháp và chỉ thị sửa đổi. |
| **Sub-2** | **Core Implementation Engineer** | `gemini-3-flash-med` | Viết mã nguồn (code) chính xác, tối ưu thuật toán, bám sát thiết kế từ Sub-1. Nếu phát hiện điểm mơ hồ, chủ động hỏi Sub-1 để làm rõ trước khi code. |
| **Sub-3** | **CI Guardian (Độc quyền CI/CD)** | `gemini-3.5-flash-lite` | **ĐỘC QUYỀN CI/CD**: Các agent khác nghiêm cấm đọc/ghi quy trình CI. Chịu trách nhiệm thực thi test suite tự động, thẩm định chất lượng pipeline và phê duyệt trạng thái xanh. |
| **Sub-4** | **QA, Security & Rigor Auditor** | `gemini-3.8-flash-high-ultra-0.2` | Rà soát lỗi logic, kiểm thử điều kiện biên (edge cases), kiểm tra rò rỉ bộ nhớ, race conditions, xung đột tiến trình và tuân thủ tiêu chuẩn chất lượng. |

---

## 2. Quy trình Thực thi & Vòng lặp Phản hồi (Execution Pipeline)

```
   [ Yêu cầu từ Người dùng ]
               │
               ▼
       ┌───────────────┐
       │     SUB-1     │  ◄── Làm rõ yêu cầu / Phân rã kiến trúc
       └───────┬───────┘
               │  Chỉ thị thiết kế & Ranh giới bài toán
               ▼
       ┌───────────────┐
       │     SUB-2     │  ◄── Hiện thực hóa mã nguồn (Code)
       └───────┬───────┘      (Hỏi Sub-1 nếu có điểm chưa rõ)
               │
               │  [CHỈ CHẠY SAU KHI SUB-2 HOÀN TẤT]
               ├─────────────────────────┐
               ▼                         ▼
       ┌───────────────┐         ┌───────────────┐
       │     SUB-3     │         │     SUB-4     │
       │  (Độc quyền)  │         │  (QA & Rigor  │
       │  Chạy Test CI │         │    Auditor)   │
       └───────┬───────┘         └───────┬───────┘
               │                         │
               └───────────┬─────────────┘
                           │
                 Có lỗi phát sinh?
                 ├── [CÓ] ──> Trao đổi với SUB-1 làm rõ nguyên nhân
                 │              │
                 │              └──> SUB-1 yêu cầu SUB-2 sửa lại code
                 │                     │
                 │                     └──> Lặp lại kiểm thử cho tới khi PASS
                 │
                 └── [KHÔNG (100% CI PASS)]
                               │
                               ▼
               [ Commit & Push lên GitHub ]
```

### Chi tiết các bước phối hợp:
1. **Giai đoạn 1 (Lập kế hoạch):** Sub-1 nhận yêu cầu, phân tích nghiệp vụ, phân tích rủi ro kỹ thuật và đưa ra tài liệu đặc tả / cấu trúc module cho Sub-2.
2. **Giai đoạn 2 (Hiện thực mã nguồn):** Sub-2 viết code theo đúng hướng dẫn của Sub-1. Trong quá trình code, nếu gặp bất kỳ điều kiện chưa rõ ràng (ambiguity), Sub-2 phải phản hồi lại Sub-1 để thống nhất, không tự ý suy đoán sai lệch đặc tả.
3. **Giai đoạn 3 (Kiểm thử song song có điều kiện):** Sub-3 và Sub-4 **CHỈ ĐƯỢC PHÉP CHẠY SAU KHI SUB-2 ĐÃ HOÀN TẤT** mã nguồn:
   - Sub-3 thực thi toàn bộ test suite trong môi trường CI cô lập.
   - Sub-4 kiểm tra tính chặt chẽ, tối ưu phần cứng, kiểm tra độ an toàn ngoại lệ và tính toàn vẹn của dữ liệu.
4. **Giai đoạn 4 (Xử lý hồi quy & Khắc phục lỗi):**
   - Nếu Sub-3 hoặc Sub-4 phát hiện bất kỳ lỗi nào (Failed Tests / Bugs / Bottlenecks):
     - Sub-3/Sub-4 lập tức tổng hợp bằng chứng, trao đổi trực tiếp với Sub-1 để làm rõ bản chất gốc rễ.
     - Sub-1 chỉ thị Sub-2 refactor hoặc sửa đổi mã nguồn.
     - Sub-2 cập nhật code và bàn giao lại cho Sub-3 & Sub-4 kiểm thử lại từ đầu.

---

## 3. Ranh giới Quyền hạn & Quy tắc Bất di bất dịch (Standing Rules)

1. **Độc quyền CI/CD:**
   - **Sub-3 nắm độc quyền CI**: Nghiêm cấm Sub-1, Sub-2, Sub-4 can thiệp, đọc, ghi hoặc chỉnh sửa các tệp cấu hình kiểm thử CI trừ khi có sự đồng thuận từ Sub-3.
2. **Nguyên tắc Commit & Push:**
   - Bắt buộc **100% test cases pass CI** trước khi tạo bất kỳ commit nào.
   - **CHỈ commit và push** khi và chỉ khi Sub-3 xác nhận trạng thái PASS hoàn toàn.
   - **KHÔNG tự ý tạo hoặc cập nhật Pull Request (PR)** trừ khi có chỉ thị trực tiếp từ người dùng.
3. **Quản lý Tiến trình Nền (Process Cleanliness):**
   - Luôn chủ động dò tìm và **kill toàn bộ background processes** (camera handles, lingering python kernels, port bindings) trước và sau mỗi phiên thực thi, đảm bảo không chiếm dụng tài nguyên hệ thống hoặc gây xung đột cổng `8000`, `CAP_DSHOW`.
4. **Định danh Tác giả (Git Identity):**
   - Mọi commit đều phải được ghi nhận dưới danh tính tác giả chuẩn xác:
     ```bash
     git config user.name "Sin Nguyen Sun"
     git config user.email "me@shyne.page"
     ```
