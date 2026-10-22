// Tests loading a binary model from Config::file_offset: models with other
// data before and after them, as when they are a member of an archive.

#include "model.hh"

#include "enumerate_vocab.hh"
#include "../util/mmap.hh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#define BOOST_TEST_MODULE BinaryOffsetTest
#include <boost/test/unit_test.hpp>

namespace lm {
namespace ngram {
namespace {

const char *TestLocation() {
  if (boost::unit_test::framework::master_test_suite().argc < 2) {
    return "test.arpa";
  }
  return boost::unit_test::framework::master_test_suite().argv[1];
}

std::string ReadFile(const std::string &name) {
  std::ifstream in(name.c_str(), std::ios::binary);
  BOOST_REQUIRE(in);
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

void WriteFile(const std::string &name, const std::string &data) {
  std::ofstream out(name.c_str(), std::ios::binary);
  out.write(data.data(), data.size());
  BOOST_REQUIRE(out);
}

// Everything in a vocabulary, in order.
class CollectVocab : public EnumerateVocab {
  public:
    void Add(WordIndex index, const StringPiece &str) {
      BOOST_CHECK_EQUAL(words.size(), static_cast<std::size_t>(index));
      words.push_back(str.as_string());
    }
    std::vector<std::string> words;
};

// The score of a sentence from the start, and from each word as the context, so that
// every part of the model is used.
template <class M> double Score(const M &model) {
  const char *sentence[] = {"looking", "on", "a", "little", "the", "more", ",", "loin", "of", "a", "<unk>", "word"};
  double total = 0.0;
  typename M::State state(model.BeginSentenceState()), out;
  for (std::size_t i = 0; i < sizeof(sentence) / sizeof(*sentence); ++i) {
    total += model.FullScore(state, model.GetVocabulary().Index(sentence[i]), out).prob;
    state = out;
  }
  return total + model.FullScore(state, model.GetVocabulary().EndSentence(), out).prob;
}

std::string Padding(std::size_t size, char fill) { return std::string(size, fill); }

template <class M> void OffsetTest() {
  const std::string plain = "offset_test_plain.binary", embedded = "offset_test_embedded.binary";

  // Make the binary.
  Config build;
  build.messages = NULL;
  build.write_mmap = plain.c_str();
  { M built(TestLocation(), build); }

  CollectVocab expected_vocab;
  Config load;
  load.messages = NULL;
  load.enumerate_vocab = &expected_vocab;
  M reference(plain.c_str(), load);
  const double expected = Score(reference);
  BOOST_CHECK_LT(expected, 0.0);

  const std::string binary = ReadFile(plain);
  const std::size_t page = util::SizePage();

  const std::size_t offsets[] = {page, 3 * page, 16 * page};
  const std::string suffixes[] = {"", "a trailer after the model", Padding(3 * page + 17, 'x'), std::string("with\0nulls\0", 11)};
  const util::LoadMethod methods[] = {util::LAZY, util::POPULATE_OR_READ, util::READ};
  for (std::size_t o = 0; o < sizeof(offsets) / sizeof(*offsets); ++o) {
    for (std::size_t s = 0; s < sizeof(suffixes) / sizeof(*suffixes); ++s) {
      WriteFile(embedded, Padding(offsets[o], '\1') + binary + suffixes[s]);
      for (std::size_t m = 0; m < sizeof(methods) / sizeof(*methods); ++m) {
        CollectVocab vocab;
        Config config;
        config.messages = NULL;
        config.file_offset = offsets[o];
        config.load_method = methods[m];
        config.enumerate_vocab = &vocab;
        M model(embedded.c_str(), config);
        BOOST_CHECK_EQUAL(expected, Score(model));
        // The vocabulary has to stop at the end of the model, not the end of the file.
        BOOST_CHECK(expected_vocab.words == vocab.words);
        ModelType type;
        BOOST_CHECK(RecognizeBinary(embedded.c_str(), type, offsets[o]));
        BOOST_CHECK_EQUAL(M::kModelType, type);
        // And without asking for the vocabulary, as a server would not.
        config.enumerate_vocab = NULL;
        M quiet(embedded.c_str(), config);
        BOOST_CHECK_EQUAL(expected, Score(quiet));
      }
    }
  }

  // The same through the virtual interface, which finds the model type itself.
  {
    WriteFile(embedded, Padding(2 * page, '\2') + binary + "trailer");
    Config config;
    config.messages = NULL;
    config.file_offset = 2 * page;
    base::Model *model = LoadVirtual(embedded.c_str(), config);
    BOOST_CHECK(model != NULL);
    delete model;
  }

  // Not at the offset: there is no model there.
  WriteFile(embedded, Padding(2 * page, '\3') + binary);
  {
    Config config;
    config.messages = NULL;
    config.file_offset = page;
    BOOST_CHECK_THROW(M(embedded.c_str(), config), util::Exception);
    config.file_offset = 100 * page;
    BOOST_CHECK_THROW(M(embedded.c_str(), config), util::Exception);
  }

  // A model cut short.
  WriteFile(embedded, Padding(page, '\4') + binary.substr(0, binary.size() - 1000));
  {
    Config config;
    config.messages = NULL;
    config.file_offset = page;
    BOOST_CHECK_THROW(M(embedded.c_str(), config), util::Exception);
  }

  // An offset that is not a page boundary cannot be mapped.  It is an error rather than a quiet
  // copy of the whole model into memory, but reading is fine.
  WriteFile(embedded, Padding(page + 512, '\5') + binary);
  {
    Config config;
    config.messages = NULL;
    config.file_offset = page + 512;
    config.load_method = util::LAZY;
    BOOST_CHECK_THROW(M(embedded.c_str(), config), util::Exception);
    config.load_method = util::POPULATE_OR_READ;
    BOOST_CHECK_THROW(M(embedded.c_str(), config), util::Exception);
    config.load_method = util::READ;
    M model(embedded.c_str(), config);
    BOOST_CHECK_EQUAL(expected, Score(model));
  }

  // Only binary models can be loaded from an offset.
  {
    Config config;
    config.messages = NULL;
    config.file_offset = page;
    BOOST_CHECK_THROW(M(TestLocation(), config), util::Exception);
  }

  std::remove(plain.c_str());
  std::remove(embedded.c_str());
}

BOOST_AUTO_TEST_CASE(probing) { OffsetTest<ProbingModel>(); }
BOOST_AUTO_TEST_CASE(rest_probing) { OffsetTest<RestProbingModel>(); }
BOOST_AUTO_TEST_CASE(trie) { OffsetTest<TrieModel>(); }
BOOST_AUTO_TEST_CASE(quant_trie) { OffsetTest<QuantTrieModel>(); }
BOOST_AUTO_TEST_CASE(bhiksha_trie) { OffsetTest<ArrayTrieModel>(); }
BOOST_AUTO_TEST_CASE(quant_bhiksha_trie) { OffsetTest<QuantArrayTrieModel>(); }

// Binary files with nothing before them are unaffected.
BOOST_AUTO_TEST_CASE(offset_zero_is_a_plain_file) {
  const char *name = "offset_test_zero.binary";
  Config build;
  build.messages = NULL;
  build.write_mmap = name;
  { ProbingModel built(TestLocation(), build); }
  Config config;
  config.messages = NULL;
  ProbingModel model(name, config);
  BOOST_CHECK_LT(Score(model), 0.0);
  std::remove(name);
}

} // namespace
} // namespace ngram
} // namespace lm
