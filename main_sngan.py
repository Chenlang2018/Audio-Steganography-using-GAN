import torch
import torch.nn as nn
import scipy.io.wavfile as wave
import os
import numpy as np
import random
import model_sngan
import time


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


set_seed(15)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists('./adv_fake_audio_sngan_test'):
    os.mkdir('./adv_fake_audio_sngan_test')

if not os.path.exists('./adv_model_sngan_g_test'):
    os.mkdir('./adv_model_sngan_g_test')

if not os.path.exists('./adv_model_sngan_d_test'):
    os.mkdir('./adv_model_sngan_d_test')

# read cover data
data = []
cover_dir = 'TIMIT16384'
filenames = os.listdir(cover_dir)
filenames.sort(key=lambda x: int(x[:-6]))
for fn in filenames:
    if fn.endswith('wav'):
        filepath = os.path.join(cover_dir, fn)
        f, wave_data = wave.read(filepath)
        wave_norm = (2 / 65535) * (wave_data.astype(np.float32) - 32767) + 1
        data.append(wave_norm)
data = np.array(data)
data = data.reshape((-1, 1, 16384))
data = torch.from_numpy(data)

# choose one test data for testing
one_test_data = []
f, one_wave_data = wave.read('TIMIT16384/1_0.wav')
one_wave_data = (2 / 65535) * (one_wave_data.astype(np.float32) - 32767) + 1
one_test_data.append(one_wave_data)
one_test_data = np.array(one_test_data)
one_test_data = one_test_data.reshape((-1, 1, 16384))
one_test_data = torch.from_numpy(one_test_data)


def to_audio(x):
    x = x.reshape(-1, 1, 16384)
    x = x.reshape(16384)
    x = 65535 / 2 * (x - 1) + 32767
    x = x.astype(np.int16)
    return x


def to_audio_batch(x):
    x = x.reshape(-1, 1, 16384)
    y = []
    for i in range(batch_size):
        x_i = x[i].reshape(16384)
        x_i = 65535 / 2 * (x_i - 1) + 32767
        y.append(x_i)
    y = np.array(y)
    return y


def norm_audio_batch(x):
    x = x.reshape(-1, 1, 16384)
    y = []
    for i in range(batch_size):
        x_i = x[i].reshape(16384)
        x_i = (2 / 65535) * (x_i.astype(np.float32) - 32767) + 1
        y.append(x_i)
    y = np.array(y)
    return y


# LSBM steganography
def embedding(generated_outputs):
    cover_audio = generated_outputs
    cover_audio = cover_audio.data.cpu().numpy()
    cover_audio = cover_audio.reshape(-1, 1, 16384)
    stego_audio = []
    for i in range(batch_size):
        cover = cover_audio[i].reshape(16384)
        cover = 65535 / 2 * (cover - 1) + 32767
        cover = cover.astype(np.int16)
        L = 16384
        stego = cover
        msg = np.random.randint(0, 2, L)
        msg = np.array(msg)
        k = np.random.randint(0, 2, L)
        k = np.array(k)
        for j in range(L):
            x = abs(cover[j])
            x = bin(x)
            x = x[2:]
            y = msg[j]
            if str(y) == x[-1]:
                stego[j] = cover[j]
            else:
                if k[j] == 0:
                    stego[j] = cover[j] - 1
                else:
                    stego[j] = cover[j] + 1
        stego = stego.reshape(16384)
        stego_audio.append(stego)
    stego_audio = np.array(stego_audio)
    return stego_audio


# ==================================Training=========================================
num_epoch = 100
batch_size = 32
d_learning_rate = 0.0001
g_learning_rate = 0.0001
s_learning_rate = 0.0001
label = torch.ones(15000)
gpu_ids = [0, 1, 2, 3]

data_tensor = torch.utils.data.TensorDataset(data, label)
dataloader = torch.utils.data.DataLoader(
    dataset=data_tensor,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)

discriminator = torch.nn.DataParallel(model_sngan.Discriminator().to(device), device_ids=gpu_ids)
discriminator = discriminator.cuda()
discriminator.load_state_dict(torch.load('discriminator_sngan-30.pth'))
discriminator.eval()

generator = torch.nn.DataParallel(model_sngan.Generator().to(device), device_ids=gpu_ids)
generator = generator.cuda()
generator.load_state_dict(torch.load('generator_sngan-30.pth'))
generator.eval()

steganalyzer = torch.nn.DataParallel(model_sngan.Steganalyzer().to(device), device_ids=gpu_ids)
steganalyzer = steganalyzer.cuda()
steganalyzer.load_state_dict(torch.load('steganalyzer_sngan_trained.pth'))
steganalyzer.eval()

d_criterion = nn.BCELoss().to(device)
s_criterion = nn.BCELoss().to(device)
g_criterion = nn.BCELoss().to(device)
q_criterion = nn.L1Loss().to(device)

d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=d_learning_rate)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=g_learning_rate)
print('Training started!=============================================================================================')
start = time.time()
for epoch in range(num_epoch):
    for i, (audio, _) in enumerate(dataloader):
        num_audio = audio.size(0)
        real_audio = audio.to(device)
        real_label = torch.ones(num_audio).reshape(batch_size, 1).to(device)
        fake_label = torch.zeros(num_audio).reshape(batch_size, 1).to(device)
        cover_label = torch.ones(num_audio).reshape(batch_size, 1).to(device)
        stego_label = torch.zeros(num_audio).reshape(batch_size, 1).to(device)

        # train discriminator
        real_out = discriminator(real_audio)
        fake_audio = generator(real_audio)
        fake_out = discriminator(fake_audio)

        d_loss_real = d_criterion(real_out, real_label)
        d_loss_fake = d_criterion(fake_out, fake_label)
        d_loss = d_loss_real + d_loss_fake

        # bp and optimize
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # train generator
        fake_audio = generator(real_audio)
        cover_audio = fake_audio
        stego_audio = embedding(cover_audio)
        stego_audio = torch.from_numpy(stego_audio).reshape(batch_size, 1, 16384).type(torch.FloatTensor)
        stego_audio = stego_audio.to(device)

        norm_stego_audio = norm_audio_batch(stego_audio.data.cpu().numpy())
        norm_stego_audio = torch.from_numpy(norm_stego_audio).reshape(batch_size, 1, 16384)
        norm_stego_audio = norm_stego_audio.to(device)

        cover_audio = to_audio_batch(fake_audio.data.cpu().numpy())
        cover_audio = torch.from_numpy(cover_audio).reshape(batch_size, 1, 16384)
        cover_audio = cover_audio.to(device)

        fake_output = discriminator(fake_audio)
        cover_output = steganalyzer(cover_audio)
        stego_output = steganalyzer(stego_audio)
        cover_scores = cover_output
        stego_scores = stego_output
        # s_loss
        s_loss_cover = s_criterion(cover_output, cover_label)
        s_loss_stego = s_criterion(stego_output, stego_label)
        s_loss = s_loss_cover + s_loss_stego
        # g_loss
        g_loss_fake = g_criterion(fake_output, real_label)
        g_loss_stego = g_criterion(stego_output, cover_label)
        g_loss_fake_quality = q_criterion(fake_audio, real_audio)
        g_loss_stego_quality = q_criterion(norm_stego_audio, real_audio)
        g_loss = g_loss_fake + g_loss_stego + g_loss_fake_quality + g_loss_stego_quality

        # bp and optimize
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 156 == 0:
            print('Epoch [{}/{}], s_loss: {:.6f}, g_loss: {:.6f} '
                  'S cover: {:.6f}, S stego: {:.6f}'.format(
                    epoch + 1, num_epoch, s_loss.item(), g_loss.item(),
                    cover_scores.data.mean(), stego_scores.data.mean()))

    # choose one audio for validation
    generator = generator.eval()
    one_fake_audio = generator(one_test_data)
    one_fake_audio = to_audio(one_fake_audio.data.cpu().numpy())
    wave.write('./adv_fake_audio_sngan_test/fake_audio-{}.wav'.format(epoch + 1), 16000, one_fake_audio)

    torch.save(generator.state_dict(), './adv_model_sngan_g_test/generator_sngan-{}.pth'.format(epoch + 1))
    torch.save(discriminator.state_dict(), './adv_model_sngan_d_test/discriminator_sngan-{}.pth'.format(epoch + 1))

print('Training finished!=============================================================================================')
end = time.time()
print('Elapsed training time: {:.2f}h'.format((end - start) / 3600))
