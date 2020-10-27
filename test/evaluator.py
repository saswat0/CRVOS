import os
import time
import torch
import utils


class VOSEvaluator(object):
    def __init__(self, dataset, device='cuda', save=False):
        self._dataset = dataset
        self._device = device
        self._save = save
        self._imsavehlp = utils.ImageSaveHelper()
        if dataset.__class__.__name__ == 'DAVIS17V2':
            self._sdm = utils.ReadSaveDAVISChallengeLabels()

    def read_video_part(self, video_part):
        images = video_part['images'].to(self._device)
        given_segannos = [seganno.to(self._device) if seganno is not None else None
                          for seganno in video_part['given_segannos']]
        segannos = video_part['segannos'].to(self._device) if video_part.get('segannos') is not None else None
        fnames = video_part['fnames']
        return images, given_segannos, segannos, fnames

    def evaluate_video(self, model, seqname, video_parts, output_path, save, video_seq):
        for video_part in video_parts:
            images, given_segannos, segannos, fnames = self.read_video_part(video_part)

            if video_seq == 0:
                _, _ = model(images, given_segannos, None)

            t0 = time.time()
            tracker_out, _ = model(images, given_segannos, None)
            t1 = time.time()

            if save is True:
                for idx in range(len(fnames)):
                    fpath = os.path.join(output_path, seqname, fnames[idx])
                    data = ((tracker_out['segs'][0, idx, 0, :, :].cpu().byte().numpy(), fpath), self._sdm)
                    self._imsavehlp.enqueue(data)
        return t1-t0, len(fnames)

    def evaluate(self, model, output_path):
        model.to(self._device)
        model.eval()
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        with torch.no_grad():
            tot_time, tot_frames, video_seq = 0.0, 0.0, 0
            for seqname, video_parts in self._dataset.get_video_generator():
                savepath = os.path.join(output_path, seqname)
                if not os.path.exists(savepath):
                    os.makedirs(savepath)
                time_elapsed, frames = self.evaluate_video(model, seqname, video_parts, output_path, self._save,
                                                           video_seq)
                tot_time += time_elapsed
                tot_frames += frames
                video_seq += 1

                if self._save is False:
                    print(seqname, 'FPS:{}, frames:{}, time:{}'.format(frames / time_elapsed, frames, time_elapsed))
                else:
                    print(seqname, 'saved')

            if self._save is False:
                print('\nTotal FPS:{}\n\n'.format(tot_frames/tot_time))
            else:
                print('\nTotal seq saved\n\n')

        self._imsavehlp.kill()