// sherpa-onnx/csrc/offline-tts-vits-impl.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_VITS_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_VITS_IMPL_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <chrono>

#if __ANDROID_API__ >= 9
#include <strstream>

#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "fst/extensions/far/far.h"
#include "kaldifst/csrc/kaldi-fst-io.h"
#include "kaldifst/csrc/text-normalizer.h"
#include "sherpa-onnx/csrc/jieba-lexicon.h"
#include "sherpa-onnx/csrc/lexicon.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/melo-tts-lexicon.h"
#include "sherpa-onnx/csrc/offline-tts-character-frontend.h"
#include "sherpa-onnx/csrc/offline-tts-frontend.h"
#include "sherpa-onnx/csrc/offline-tts-impl.h"
#include "sherpa-onnx/csrc/offline-tts-vits-model.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/piper-phonemize-lexicon.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OfflineTtsVitsImpl : public OfflineTtsImpl {
 public:
  explicit OfflineTtsVitsImpl(const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsVitsModel>(config.model)) {
    InitFrontend();

    if (!config.rule_fsts.empty()) {
      std::vector<std::string> files;
      SplitStringToVector(config.rule_fsts, ",", false, &files);
      tn_list_.reserve(files.size());
      for (const auto &f : files) {
        if (config.model.debug) {
          SHERPA_ONNX_LOGE("rule fst: %s", f.c_str());
        }
        tn_list_.push_back(std::make_unique<kaldifst::TextNormalizer>(f));
      }
    }

    if (!config.rule_fars.empty()) {
      if (config.model.debug) {
        SHERPA_ONNX_LOGE("Loading FST archives");
      }
      std::vector<std::string> files;
      SplitStringToVector(config.rule_fars, ",", false, &files);

      tn_list_.reserve(files.size() + tn_list_.size());

      for (const auto &f : files) {
        if (config.model.debug) {
          SHERPA_ONNX_LOGE("rule far: %s", f.c_str());
        }
        std::unique_ptr<fst::FarReader<fst::StdArc>> reader(
            fst::FarReader<fst::StdArc>::Open(f));
        for (; !reader->Done(); reader->Next()) {
          std::unique_ptr<fst::StdConstFst> r(
              fst::CastOrConvertToConstFst(reader->GetFst()->Copy()));

          tn_list_.push_back(
              std::make_unique<kaldifst::TextNormalizer>(std::move(r)));
        }
      }

      if (config.model.debug) {
        SHERPA_ONNX_LOGE("FST archives loaded!");
      }
    }
  }

#if __ANDROID_API__ >= 9
  OfflineTtsVitsImpl(AAssetManager *mgr, const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsVitsModel>(mgr, config.model)) {
    InitFrontend(mgr);

    if (!config.rule_fsts.empty()) {
      std::vector<std::string> files;
      SplitStringToVector(config.rule_fsts, ",", false, &files);
      tn_list_.reserve(files.size());
      for (const auto &f : files) {
        if (config.model.debug) {
          SHERPA_ONNX_LOGE("rule fst: %s", f.c_str());
        }
        auto buf = ReadFile(mgr, f);
        std::istrstream is(buf.data(), buf.size());
        tn_list_.push_back(std::make_unique<kaldifst::TextNormalizer>(is));
      }
    }

    if (!config.rule_fars.empty()) {
      std::vector<std::string> files;
      SplitStringToVector(config.rule_fars, ",", false, &files);
      tn_list_.reserve(files.size() + tn_list_.size());

      for (const auto &f : files) {
        if (config.model.debug) {
          SHERPA_ONNX_LOGE("rule far: %s", f.c_str());
        }

        auto buf = ReadFile(mgr, f);

        std::unique_ptr<std::istream> s(
            new std::istrstream(buf.data(), buf.size()));

        std::unique_ptr<fst::FarReader<fst::StdArc>> reader(
            fst::FarReader<fst::StdArc>::Open(std::move(s)));

        for (; !reader->Done(); reader->Next()) {
          std::unique_ptr<fst::StdConstFst> r(
              fst::CastOrConvertToConstFst(reader->GetFst()->Copy()));

          tn_list_.push_back(
              std::make_unique<kaldifst::TextNormalizer>(std::move(r)));
        }  // for (; !reader->Done(); reader->Next())
      }    // for (const auto &f : files)
    }      // if (!config.rule_fars.empty())
  }
#endif

  int32_t SampleRate() const override {
    return model_->GetMetaData().sample_rate;
  }

  int32_t NumSpeakers() const override {
    return model_->GetMetaData().num_speakers;
  }

  GeneratedAudio Generate(
      const std::string &_text, int64_t sid = 0, float speed = 1.0,
      GeneratedAudioCallback callback = nullptr) const override {
    const auto &meta_data = model_->GetMetaData();
    int32_t num_speakers = meta_data.num_speakers;

    if (num_speakers == 0 && sid != 0) {
      SHERPA_ONNX_LOGE(
          "This is a single-speaker model and supports only sid 0. Given sid: "
          "%d. sid is ignored",
          static_cast<int32_t>(sid));
    }

    if (num_speakers != 0 && (sid >= num_speakers || sid < 0)) {
      SHERPA_ONNX_LOGE(
          "This model contains only %d speakers. sid should be in the range "
          "[%d, %d]. Given: %d. Use sid=0",
          num_speakers, 0, num_speakers - 1, static_cast<int32_t>(sid));
      sid = 0;
    }

    std::string text = _text;
    if (config_.model.debug) {
      SHERPA_ONNX_LOGE("Raw text: %s", text.c_str());
    }

    if (!tn_list_.empty()) {
      for (const auto &tn : tn_list_) {
        text = tn->Normalize(text);
      }
      if (config_.model.debug) {
        SHERPA_ONNX_LOGE("After normalizing: %s", text.c_str());
      }
    }

    std::vector<TokenIDs> token_ids =
        frontend_->ConvertTextToTokenIds(text, meta_data.voice);

    if (token_ids.empty() ||
        (token_ids.size() == 1 && token_ids[0].tokens.empty())) {
      SHERPA_ONNX_LOGE("Failed to convert %s to token IDs", text.c_str());
      return {};
    }

    std::vector<std::vector<int64_t>> tokens;
    std::vector<std::vector<int64_t>> tones;
    std::vector<std::string> words;

    tokens.reserve(token_ids.size());
    tones.reserve(token_ids.size());

    for (auto &token_id : token_ids) {
        tokens.push_back(std::move(token_id.tokens));
        tones.push_back(std::move(token_id.tones));
        words.push_back(std::move(token_id.words));
    }

    // TODO(fangjun): add blank inside the frontend, not here
    if (meta_data.add_blank && config_.model.vits.data_dir.empty() &&
        meta_data.frontend != "characters") {
      for (auto &k : tokens) {
        k = AddBlank(k);
      }

      for (auto &k : tones) {
        k = AddBlank(k);
      }
    }

    int32_t size = static_cast<int32_t>(tokens.size());

    if (config_.max_num_sentences <= 0 || size <= config_.max_num_sentences) {
      auto ans = Process(tokens, tones, sid, speed);
      if (callback) {
        callback(ans.samples.data(), ans.samples.size(), 1.0);
      }
      return ans;
    }

    // the input text is too long, we process sentences within it in batches
    // to avoid OOM. Batch size is config_.max_num_sentences
    std::vector<std::vector<int64_t>> batch_tokens;
    std::vector<std::vector<int64_t>> batch_tones;
    std::string batch_words;

    int32_t batch_size = config_.max_num_sentences;
    batch_tokens.reserve(batch_size);
    batch_tones.reserve(batch_size);
    int32_t batch_count = size / batch_size;

    int32_t should_continue = 1;
    int32_t index = 0;
    GeneratedAudio ans;


    SHERPA_ONNX_LOGE("FGGG Process start ---------------");
    for (int m = 0; m <= batch_count && should_continue; ++m) {
      batch_tokens.clear();
      batch_tones.clear();
      batch_words.clear();

      for (int n = 0; n < batch_size && index < size; ++n, ++index) {
        batch_words += words[index];
        batch_tokens.push_back(std::move(tokens[index]));
        if (!tones.empty()) {
          batch_tones.push_back(std::move(tones[index]));
        }
      }
      if (!batch_tokens.empty()) {
        SHERPA_ONNX_LOGE("FGGGG Process .size = %zu .words = %s", batch_tokens.size(), batch_words.c_str());
        auto start = std::chrono::high_resolution_clock::now();
        auto [samples, sample_rate] = Process(batch_tokens, batch_tones, sid, speed);
        ans.sample_rate = sample_rate;
        ans.samples.insert(ans.samples.end(), samples.begin(), samples.end());
        if (callback) {
          auto length = static_cast<double>(samples.size()) * 1000 / 44100;
          should_continue = callback(samples.data(), samples.size(), index * 100.0 / size);
          auto end = std::chrono::high_resolution_clock::now();
          auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
          SHERPA_ONNX_LOGE("FGGGG Processed play = %lld process = %lld .words = %s", static_cast<long>(length), duration.count(), batch_words.c_str());
        }
      }
    }
    SHERPA_ONNX_LOGE("FGGGG Process end ---------------");

    return ans;
  }

 private:
#if __ANDROID_API__ >= 9
  void InitFrontend(AAssetManager *mgr) {
    const auto &meta_data = model_->GetMetaData();

    if (meta_data.frontend == "characters") {
      frontend_ = std::make_unique<OfflineTtsCharacterFrontend>(
          mgr, config_.model.vits.tokens, meta_data);
    } else if ((meta_data.is_piper || meta_data.is_coqui ||
                meta_data.is_icefall) &&
               !config_.model.vits.data_dir.empty()) {
      frontend_ = std::make_unique<PiperPhonemizeLexicon>(
          mgr, config_.model.vits.tokens, config_.model.vits.data_dir,
          meta_data);
    } else {
      if (config_.model.vits.lexicon.empty()) {
        SHERPA_ONNX_LOGE(
            "Not a model using characters as modeling unit. Please provide "
            "--vits-lexicon if you leave --vits-data-dir empty");
        exit(-1);
      }

      frontend_ = std::make_unique<Lexicon>(
          mgr, config_.model.vits.lexicon, config_.model.vits.tokens,
          meta_data.punctuations, meta_data.language, config_.model.debug);
    }
  }
#endif

  void InitFrontend() {
    const auto &meta_data = model_->GetMetaData();

    if (meta_data.jieba && config_.model.vits.dict_dir.empty()) {
      SHERPA_ONNX_LOGE(
          "Please provide --vits-dict-dir for Chinese TTS models using jieba");
      exit(-1);
    }

    if (!meta_data.jieba && !config_.model.vits.dict_dir.empty()) {
      SHERPA_ONNX_LOGE(
          "Current model is not using jieba but you provided --vits-dict-dir");
      exit(-1);
    }

    if (meta_data.frontend == "characters") {
      frontend_ = std::make_unique<OfflineTtsCharacterFrontend>(
          config_.model.vits.tokens, meta_data);
    } else if (meta_data.jieba && !config_.model.vits.dict_dir.empty() &&
               meta_data.is_melo_tts) {
      frontend_ = std::make_unique<MeloTtsLexicon>(
          config_.model.vits.lexicon, config_.model.vits.tokens,
          config_.model.vits.dict_dir, model_->GetMetaData(),
          config_.model.debug);
    } else if (meta_data.jieba && !config_.model.vits.dict_dir.empty()) {
      frontend_ = std::make_unique<JiebaLexicon>(
          config_.model.vits.lexicon, config_.model.vits.tokens,
          config_.model.vits.dict_dir, model_->GetMetaData(),
          config_.model.debug);
    } else if ((meta_data.is_piper || meta_data.is_coqui ||
                meta_data.is_icefall) &&
               !config_.model.vits.data_dir.empty()) {
      frontend_ = std::make_unique<PiperPhonemizeLexicon>(
          config_.model.vits.tokens, config_.model.vits.data_dir,
          model_->GetMetaData());
    } else {
      if (config_.model.vits.lexicon.empty()) {
        SHERPA_ONNX_LOGE(
            "Not a model using characters as modeling unit. Please provide "
            "--vits-lexicon if you leave --vits-data-dir empty");
        exit(-1);
      }
      frontend_ = std::make_unique<Lexicon>(
          config_.model.vits.lexicon, config_.model.vits.tokens,
          meta_data.punctuations, meta_data.language, config_.model.debug);
    }
  }

  std::vector<int64_t> AddBlank(const std::vector<int64_t> &x) const {
    // we assume the blank ID is 0
    std::vector<int64_t> buffer(x.size() * 2 + 1);
    int32_t i = 1;
    for (auto k : x) {
      buffer[i] = k;
      i += 2;
    }
    return buffer;
  }

  GeneratedAudio Process(const std::vector<std::vector<int64_t>> &tokens,
                         const std::vector<std::vector<int64_t>> &tones,
                         int32_t sid, float speed) const {
    int32_t num_tokens = 0;
    for (const auto &k : tokens) {
      num_tokens += k.size();
    }

    std::vector<int64_t> x;
    x.reserve(num_tokens);
    for (const auto &k : tokens) {
      x.insert(x.end(), k.begin(), k.end());
    }

    std::vector<int64_t> tone_list;
    if (!tones.empty()) {
      tone_list.reserve(num_tokens);
      for (const auto &k : tones) {
        tone_list.insert(tone_list.end(), k.begin(), k.end());
      }
    }

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 2> x_shape = {1, static_cast<int32_t>(x.size())};
    Ort::Value x_tensor = Ort::Value::CreateTensor(
        memory_info, x.data(), x.size(), x_shape.data(), x_shape.size());

    Ort::Value tones_tensor{nullptr};
    if (!tones.empty()) {
      tones_tensor = Ort::Value::CreateTensor(memory_info, tone_list.data(),
                                              tone_list.size(), x_shape.data(),
                                              x_shape.size());
    }

    Ort::Value audio{nullptr};
    if (tones.empty()) {
      audio = model_->Run(std::move(x_tensor), sid, speed);
    } else {
      audio =
          model_->Run(std::move(x_tensor), std::move(tones_tensor), sid, speed);
    }

    std::vector<int64_t> audio_shape = audio.GetTensorTypeAndShapeInfo().GetShape();

    int64_t total = 1;
    // The output shape may be (1, 1, total) or (1, total) or (total,)
    for (auto i : audio_shape) {
      total *= i;
    }

    const float *p = audio.GetTensorData<float>();

    GeneratedAudio ans;
    ans.sample_rate = model_->GetMetaData().sample_rate;
    ans.samples = std::vector<float>(p, p + total);
    return ans;
  }

 private:
  OfflineTtsConfig config_;
  std::unique_ptr<OfflineTtsVitsModel> model_;
  std::vector<std::unique_ptr<kaldifst::TextNormalizer>> tn_list_;
  std::unique_ptr<OfflineTtsFrontend> frontend_;
};

}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_VITS_IMPL_H_
