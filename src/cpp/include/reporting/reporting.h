//
// Created by Jason Mohoney on 8/24/21.
//

#ifndef MARIUS_SRC_CPP_INCLUDE_REPORTING_H_
#define MARIUS_SRC_CPP_INCLUDE_REPORTING_H_

#include "common/datatypes.h"
#include "common/pybind_headers.h"
#include "data/batch.h"
#include "pipeline/pipeline_constants.h"
#include "sys/types.h"
#include "sys/sysinfo.h"

using pyobj = pybind11::object;

class Metric {
  public:
    std::string name_;
    std::string unit_;

    virtual ~Metric() {};
};

class RankingMetric : public Metric {
  public:
    virtual torch::Tensor computeMetric(torch::Tensor ranks) = 0;
};

class HitskMetric : public RankingMetric {
    int k_;
  public:
    HitskMetric(int k);

    torch::Tensor computeMetric(torch::Tensor ranks);
};

class MeanRankMetric : public RankingMetric {
  public:
    MeanRankMetric();

    torch::Tensor computeMetric(torch::Tensor ranks);
};

class MeanReciprocalRankMetric : public RankingMetric {
  public:
    MeanReciprocalRankMetric();

    torch::Tensor computeMetric(torch::Tensor ranks);
};

class ClassificationMetric : public Metric {
  public:
    virtual torch::Tensor computeMetric(torch::Tensor y_true, torch::Tensor y_pred) = 0;
};

class CategoricalAccuracyMetric : public ClassificationMetric {
  public:
    CategoricalAccuracyMetric();

    torch::Tensor computeMetric(torch::Tensor y_true, torch::Tensor y_pred) override;
};

class Reporter {
  private:
    std::mutex *lock_;
  public:
    std::vector<shared_ptr<Metric>> metrics_;

    Reporter() {
        lock_ = new std::mutex();
    }

    virtual ~Reporter();

    void lock() {
        lock_->lock();
    }

    void unlock() {
        lock_->unlock();
    }

    void addMetric(shared_ptr<Metric> metric) {
        metrics_.emplace_back(metric);
    }

    // void reportToFile(std::string output_filename) {

    // }

    virtual void report() = 0;

};

class LinkPredictionReporter : public Reporter {
  public:
    std::vector<torch::Tensor> per_batch_ranks_;
    std::vector<torch::Tensor> per_batch_scores_;
    std::vector<torch::Tensor> per_batch_edges_;
    torch::Tensor all_ranks_;
    torch::Tensor all_scores_;
    torch::Tensor all_edges_;

    LinkPredictionReporter();

    ~LinkPredictionReporter();

    void clear();

    torch::Tensor computeRanks(torch::Tensor pos_scores, torch::Tensor neg_scores);

    void addResult(torch::Tensor pos_scores, torch::Tensor neg_scores, torch::Tensor edges = torch::Tensor());

    void report() override;

    // void reportToFile(std::string output_filename);

    void save(string directory, bool scores, bool ranks);
};

class NodeClassificationReporter : public Reporter {

public:


    std::vector<torch::Tensor> per_batch_y_true_;
    std::vector<torch::Tensor> per_batch_y_pred_;
    std::vector<torch::Tensor> per_batch_nodes_;
    torch::Tensor all_y_true_;
    torch::Tensor all_y_pred_;
    torch::Tensor all_nodes_;

    NodeClassificationReporter();

    ~NodeClassificationReporter();

    void clear();

    void addResult(torch::Tensor y_true, torch::Tensor y_pred, torch::Tensor node_ids = torch::Tensor());

    void report() override;

    void save(string directory, bool labels);
};

class ProgressReporter : public Reporter {
    std::string item_name_;
    int64_t total_items_;
    int64_t current_item_;
    int total_reports_;
    int64_t next_report_;
    int64_t items_per_report_;
  public:
    ProgressReporter(std::string item_name, int64_t total_items, int total_reports);

    ~ProgressReporter();

    void clear();

    void addResult(int64_t items_processed);

    void report() override;


};

class BatchTimingReporter : public Reporter {
    std::vector<BatchTiming> times_;
    pyobj tensorboard_converter_module_;
  public:
    pyobj summary_writer_;
    std::string output_filename_;

    BatchTimingReporter();

    ~BatchTimingReporter() {clear();}

    void clear() {times_ = {}; py::finalize_interpreter();}

    void setupSummaryWriter(std::string log_directory);

    void addResult(BatchTiming batch_timing);

    void appendBatchTimingResult(BatchTiming batch_timing);

    void report() override;
};

class MemorySampler {
protected:
    std::atomic<bool> done_;
    struct timespec sampling_interval_;
    std::thread thread_;
    std::vector<std::vector<float>> cpu_mem_samples_;
    std::vector<float> sub_cpu_mem_samples_;
    std::vector<std::vector<float>> gpu_mem_samples_;
    std::vector<float> sub_gpu_mem_samples_;
public:
    explicit MemorySampler();

    void run();

    void spawn() {
        thread_ = std::thread(&MemorySampler::run, this);
    }

    void start() {
        spawn();
        done_ = false;
        sub_cpu_mem_samples_ = {};
        sub_gpu_mem_samples_ = {};
    }

    void stop(std::string output_filename) {
        done_ = true;
        cpu_mem_samples_.emplace_back(sub_cpu_mem_samples_);
        if (thread_.joinable()){
            thread_.join();
        }
        report(output_filename);
    }

    void clear() {};

    void report(std::string output_filename);
};

#endif //MARIUS_SRC_CPP_INCLUDE_REPORTING_H_
