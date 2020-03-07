import nltk
from utils import *
import pickle

dataset = '102flowers'  #
need_256 = True  # set to True for stackGAN
if dataset == '102flowers':
    """
    images.shape = [8000, 64, 64, 3]
    captions_ids = [80000, any]
    """
    cwd = os.getcwd()
    img_dir = os.path.join(cwd, '102flowers')
    caption_dir = os.path.join(cwd, 'text_c10')
    VOC_FIR = cwd + '/vocab.txt'
    ## load captions
    caption_sub_dir = load_folder_list(caption_dir)
    captions_dict = {}
    processed_capts = []
    for sub_dir in caption_sub_dir:  # get caption file list
        with tl.ops.suppress_stdout():
            files = tl.files.load_file_list(path=sub_dir, regx='^image_[0-9]+\.txt')
            for i, f in enumerate(files):
                file_dir = os.path.join(sub_dir, f)
                key = int(re.findall('\d+', f)[0])
                t = open(file_dir, 'r')
                lines = []
                for line in t:
                    line = preprocess_caption(line)
                    lines.append(line)
                    processed_capts.append(tl.nlp.process_sentence(line, start_word="<S>", end_word="</S>"))
                assert len(lines) == 10, "Every flower image have 10 captions"
                captions_dict[key] = lines
    print(" * %d x %d captions found " % (len(captions_dict), len(lines)))
    ## build vocab
    if not os.path.isfile('vocab.txt'):
        _ = tl.nlp.create_vocab(processed_capts, word_counts_output_file=VOC_FIR, min_word_count=1)
    else:
        print("WARNING: vocab.txt already exists")
    vocab = tl.nlp.Vocabulary(VOC_FIR, start_word="<S>", end_word="</S>", unk_word="<UNK>")
    ## store all captions ids in list
    captions_ids = []
    try:  # python3
        tmp = captions_dict.items()
    except:  # python3
        tmp = captions_dict.iteritems()
    for key, value in tmp:
        for v in value:
            captions_ids.append(
                [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(v)] + [vocab.end_id])  # add END_ID
    captions_ids = np.asarray(captions_ids)
    print(" * tokenized %d captions" % len(captions_ids))

    ## check
    img_capt = captions_dict[1][1]
    print("img_capt: %s" % img_capt)
    print("nltk.tokenize.word_tokenize(img_capt): %s" % nltk.tokenize.word_tokenize(img_capt))
    img_capt_ids = [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(img_capt)]  # img_capt.split(' ')]
    print("img_capt_ids: %s" % img_capt_ids)
    print("id_to_word: %s" % [vocab.id_to_word(id) for id in img_capt_ids])

    ## load images
    with tl.ops.suppress_stdout():  # get image files list
        imgs_title_list = sorted(tl.files.load_file_list(path=img_dir, regx='^image_[0-9]+\.jpg'))
    print(" * %d images found, start loading and resizing ..." % len(imgs_title_list))
    s = time.time()
    images = []
    images_256 = []
    for name in imgs_title_list:
        img_raw = imageio.imread(os.path.join(img_dir, name))
        img = tl.prepro.imresize(img_raw, size=[64, 64])  # (64, 64, 3)
        img = img.astype(np.float32)
        images.append(img)
        if need_256:
            img = tl.prepro.imresize(img_raw, size=[256, 256])  # (256, 256, 3)
            img = img.astype(np.float32)
            images_256.append(img)
    print(" * loading and resizing took %ss" % (time.time() - s))

    n_images = len(captions_dict)
    n_captions = len(captions_ids)
    n_captions_per_image = len(lines)  # 10

    print("n_captions: %d n_images: %d n_captions_per_image: %d" % (n_captions, n_images, n_captions_per_image))

    captions_ids_train, captions_ids_test = captions_ids[: 8000 * n_captions_per_image], captions_ids[
                                                                                         8000 * n_captions_per_image:]
    images_train, images_test = images[:8000], images[8000:]
    if need_256:
        images_train_256, images_test_256 = images_256[:8000], images_256[8000:]
    n_images_train = len(images_train)
    n_images_test = len(images_test)
    n_captions_train = len(captions_ids_train)
    n_captions_test = len(captions_ids_test)
    print("n_images_train:%d n_captions_train:%d" % (n_images_train, n_captions_train))
    print("n_images_test:%d  n_captions_test:%d" % (n_images_test, n_captions_test))


def save_all(targets, file):
    with open(file, 'wb') as f:
        pickle.dump(targets, f)


save_all(vocab, '_vocab.pickle')
save_all((images_train_256, images_train), '_image_train.pickle')
save_all((images_test_256, images_test), '_image_test.pickle')
save_all((n_captions_train, n_captions_test, n_captions_per_image, n_images_train, n_images_test), '_n.pickle')
save_all((captions_ids_train, captions_ids_test), '_caption.pickle')
