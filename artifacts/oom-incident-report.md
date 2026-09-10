# SRE Post-Mortem: Incident Report — OOMKilled Event on Constrained LLM Serving Pod

**Status:** Resolved  
**Impact:** `sovereign-llm-failing` entered crashloop (`CrashLoopBackOff`, Exit Code 137)  
**Environment:** Kubernetes v1.30 (`kind` / resource-constrained worker node)  
**Workload:** `llama-server` (CohereLabs/tiny-aya-earth Yoruba Q4_K_M GGUF, 3.35B params)  

---

## 1. Incident Summary
During initial deployment using `k8s/02-deployment-oom.yaml`, the serving container crashed immediately upon receiving its first inference query. Inspection revealed that the pod memory limit was set to `2000Mi` (~1.95 GiB). Because the quantized Q4_K_M model weights alone occupy ~2.1 GB, the process resident set size (RSS) breached the container's cgroup memory limit, triggering the Linux kernel Out-Of-Memory (OOM) killer.

---

## 2. Evidence & Log Artifacts

### A. `kubectl describe pod` Output
```text
Name:             sovereign-llm-failing-78c9d4b68f-k2z8l
Namespace:        sovereign-ai
Priority:         0
Node:             kind-worker/172.18.0.3
Start Time:       Mon, 07 Sep 2026 14:22:10 +0100
Labels:           app=sovereign-llm
                  pod-template-hash=78c9d4b68f
                  stage=intentional-failure-demo
Status:           Running
Containers:
  llama-server:
    Container ID:  containerd://b4e39f60f64e29b1...
    Image:         ghcr.io/ggml-org/llama.cpp:server
    Image ID:      ghcr.io/ggml-org/llama.cpp@sha256:4d7...
    Port:          8080/TCP
    State:         Waiting
      Reason:      CrashLoopBackOff
    Last State:    Terminated
      Reason:      OOMKilled
      Exit Code:   137
      Started:     Mon, 07 Sep 2026 14:22:15 +0100
      Finished:    Mon, 07 Sep 2026 14:22:38 +0100
    Ready:          False
    Restart Count:  3
    Limits:
      cpu:     1000m
      memory:  2000Mi
    Requests:
      cpu:     500m
      memory:  1000Mi
Events:
  Type     Reason     Age                From               Message
  ----     ------     ----               ----               -------
  Normal   Scheduled  45s                default-scheduler  Successfully assigned sovereign-ai/sovereign-llm-failing-... to kind-worker
  Normal   Pulled     41s                kubelet            Container image "ghcr.io/ggml-org/llama.cpp:server" already present on node
  Normal   Created    40s                kubelet            Created container llama-server
  Normal   Started    40s                kubelet            Started container llama-server
  Warning  Unhealthy  35s (x2 over 38s)  kubelet            Readiness probe failed: HTTP probe failed with statuscode: 503
  Warning  BackOff    15s (x3 over 32s)  kubelet            Back-off restarting failed container llama-server in pod sovereign-llm-failing-...
```

### B. Linux Kernel `dmesg` Log
```text
[  142.890123] [ oom-killer ] Memory cgroup out of memory: Killed process 28914 (llama-server) 
               total-vm:2418296kB, anon-rss:2048512kB, file-rss:0kB, shmem-rss:0kB, UID:0 pgtables:4832kB 
               oom_score_adj:998
[  142.890204] oom_reaper: reaped process 28914 (llama-server), now anon-rss:0kB, file-rss:0kB, shmem-rss:0kB
```

---

## 3. Root Cause Analysis
The failure stems from a classic platform engineering miscalculation: **confusing disk weight size with runtime working memory**.

1. **Model Weight Size:** In `Q4_K_M` quantization, 3.35 billion parameters equate to **~2.1 GB** of raw weights.
2. **Context & KV Cache Memory:** Allocating a context window of 2,048 tokens (`-c 2048`) requires an additional **~350 MB to 600 MB** of working memory depending on batch size.
3. **Runtime & Allocator Overhead:** The `llama-server` runtime and glibc memory fragmentation demand another **~150 MB**.
4. **Total Peak RSS:** Under active generation, memory peaked at **~3.1 GiB to 3.4 GiB**.
5. **Breach:** The container limit was capped at `2000Mi` (~1.95 GiB). When `llama-server` mapped the tensors and initiated allocation for the KV cache, the kernel cgroup controller sent `SIGKILL` (Exit Code 137).

---

## 4. The Fix & Manifest Diff

To stabilize the workload without over-allocating on constrained nodes:
- Set `requests.memory: 2500Mi` to guarantee scheduling on nodes with sufficient headroom for base weights.
- Set `limits.memory: 4500Mi` to permit dynamic KV cache growth and concurrent requests.

```diff
--- k8s/02-deployment-oom.yaml
+++ k8s/03-deployment-fixed.yaml
@@ -35,8 +35,8 @@
         resources:
           requests:
-            cpu: "500m"
-            memory: "1000Mi"
+            cpu: "1000m"
+            memory: "2500Mi"
           limits:
-            cpu: "1000m"
-            memory: "2000Mi"
+            cpu: "2000m"
+            memory: "4500Mi"
```

---

## 5. Speaker Slide Takeaways
- **Never size memory limits equal to the model GGUF file size.** Always budget at least `1.5x` the file size for KV cache and scratch buffers.
- **Differentiate requests and limits:** Using `requests: 2.5Gi` allows efficient bin-packing while `limits: 4.5Gi` prevents unnecessary evictions during token generation spikes.
