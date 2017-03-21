import os
import argparse
import shutil
import glob

TOKEN_VOCAB = [
  'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b',
  'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en',
  'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv',
  'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng',
  'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh',
  't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z',
  'zh'
]

SAMPLE_RATE = 16000.0

TRAIN_SPEAKERS = [
  'faem0', 'fajw0', 'falk0', 'falr0', 'fapb0', 'fbas0', 'fbcg1', 'fbch0', 'fbjl0', 'fblv0',
  'fbmh0', 'fbmj0', 'fcag0', 'fcaj0', 'fcdr1', 'fceg0', 'fcjf0', 'fcjs0', 'fcke0', 'fclt0',
  'fcmg0', 'fcmm0', 'fcrz0', 'fcyl0', 'fdas1', 'fdaw0', 'fdfb0', 'fdjh0', 'fdkn0', 'fdml0',
  'fdmy0', 'fdnc0', 'fdtd0', 'fdxw0', 'feac0', 'fear0', 'fecd0', 'feeh0', 'feme0', 'fetb0',
  'fexm0', 'fgcs0', 'fgdp0', 'fgmb0', 'fgrw0', 'fhlm0', 'fhxs0', 'fjdm2', 'fjen0', 'fjhk0',
  'fjkl0', 'fjlg0', 'fjlr0', 'fjrb0', 'fjrp1', 'fjsk0', 'fjsp0', 'fjwb1', 'fjxm0', 'fjxp0',
  'fkaa0', 'fkde0', 'fkdw0', 'fkfb0', 'fkkh0', 'fklc0', 'fklc1', 'fklh0', 'fksr0', 'flac0',
  'flag0', 'fleh0', 'flet0', 'flhd0', 'flja0', 'fljd0', 'fljg0', 'flkm0', 'flma0', 'flmc0',
  'flmk0', 'flod0', 'fltm0', 'fmah1', 'fmbg0', 'fmem0', 'fmjb0', 'fmjf0', 'fmju0', 'fmkc0',
  'fmkf0', 'fmmh0', 'fmpg0', 'fnkl0', 'fntb0', 'fpab1', 'fpac0', 'fpad0', 'fpaf0', 'fpaz0',
  'fpjf0', 'fpls0', 'fpmy0', 'freh0', 'frjb0', 'frll0', 'fsag0', 'fsah0', 'fsak0', 'fsbk0',
  'fscn0', 'fsdc0', 'fsdj0', 'fsgf0', 'fsjg0', 'fsjk1', 'fsjs0', 'fsjw0', 'fskc0', 'fskl0',
  'fskp0', 'fsls0', 'fsma0', 'fsmm0', 'fsms1', 'fspm0', 'fsrh0', 'fssb0', 'ftaj0', 'ftbr0',
  'ftbw0', 'ftlg0', 'ftmg0', 'fvfb0', 'fvkb0', 'fvmh0', 'mabc0', 'madc0', 'madd0', 'maeb0',
  'maeo0', 'mafm0', 'majp0', 'makb0', 'makr0', 'mapv0', 'marc0', 'marw0', 'mbar0', 'mbbr0',
  'mbcg0', 'mbef0', 'mbgt0', 'mbjv0', 'mbma0', 'mbma1', 'mbml0', 'mbom0', 'mbsb0', 'mbth0',
  'mbwp0', 'mcae0', 'mcal0', 'mcdc0', 'mcdd0', 'mcdr0', 'mcef0', 'mcew0', 'mchl0', 'mclk0',
  'mclm0', 'mcpm0', 'mcre0', 'mcss0', 'mcth0', 'mctm0', 'mcxm0', 'mdac0', 'mdas0', 'mdbb1',
  'mdbp0', 'mdcd0', 'mdcm0', 'mddc0', 'mded0', 'mdef0', 'mdem0', 'mdhl0', 'mdhs0', 'mdjm0',
  'mdks0', 'mdlb0', 'mdlc0', 'mdlc1', 'mdlc2', 'mdlh0', 'mdlm0', 'mdlr0', 'mdlr1', 'mdma0',
  'mdmt0', 'mdns0', 'mdpb0', 'mdpk0', 'mdps0', 'mdrd0', 'mdsj0', 'mdss0', 'mdss1', 'mdtb0',
  'mdwd0', 'mdwh0', 'mdwm0', 'meal0', 'medr0', 'mefg0', 'megj0', 'mejl0', 'mejs0', 'mesg0',
  'mesj0', 'mewm0', 'mfer0', 'mfmc0', 'mfrm0', 'mfwk0', 'mfxs0', 'mfxv0', 'mgaf0', 'mgag0',
  'mgak0', 'mgar0', 'mgaw0', 'mges0', 'mgjc0', 'mgrl0', 'mgrp0', 'mgsh0', 'mgsl0', 'mgxp0',
  'mhbs0', 'mhit0', 'mhjb0', 'mhmg0', 'mhmr0', 'mhrm0', 'mhxl0', 'milb0', 'mjac0', 'mjae0',
  'mjai0', 'mjbg0', 'mjda0', 'mjdc0', 'mjde0', 'mjdg0', 'mjdm0', 'mjeb0', 'mjeb1', 'mjee0',
  'mjfh0', 'mjfr0', 'mjhi0', 'mjjb0', 'mjjj0', 'mjjm0', 'mjkr0', 'mjlb0', 'mjlg1', 'mjls0',
  'mjma0', 'mjmd0', 'mjmm0', 'mjpg0', 'mjpm0', 'mjpm1', 'mjra0', 'mjrg0', 'mjrh0', 'mjrh1',
  'mjrk0', 'mjrp0', 'mjsr0', 'mjwg0', 'mjws0', 'mjwt0', 'mjxa0', 'mjxl0', 'mkag0', 'mkah0',
  'mkaj0', 'mkam0', 'mkdb0', 'mkdd0', 'mkdt0', 'mkes0', 'mkjo0', 'mkln0', 'mklr0', 'mkls0',
  'mkls1', 'mklw0', 'mkrg0', 'mkxl0', 'mlbc0', 'mlel0', 'mljc0', 'mljh0', 'mlns0', 'mlsh0',
  'mmaa0', 'mmab1', 'mmag0', 'mmam0', 'mmar0', 'mmbs0', 'mmcc0', 'mmdb0', 'mmdg0', 'mmdm0',
  'mmdm1', 'mmds0', 'mmea0', 'mmeb0', 'mmgc0', 'mmgg0', 'mmgk0', 'mmjb1', 'mmlm0', 'mmpm0',
  'mmrp0', 'mmsm0', 'mmvp0', 'mmwb0', 'mmws0', 'mmws1', 'mmxs0', 'mnet0', 'mntw0', 'mpar0',
  'mpeb0', 'mpfu0', 'mpgh0', 'mpgr0', 'mpgr1', 'mpmb0', 'mppc0', 'mprb0', 'mprd0', 'mprk0',
  'mprt0', 'mpsw0', 'mrab0', 'mrab1', 'mrai0', 'mram0', 'mrav0', 'mrbc0', 'mrcg0', 'mrcw0',
  'mrdd0', 'mrdm0', 'mrds0', 'mree0', 'mreh1', 'mrem0', 'mrew1', 'mrfk0', 'mrfl0', 'mrgm0',
  'mrgs0', 'mrhl0', 'mrjb1', 'mrjh0', 'mrjm0', 'mrjm1', 'mrjt0', 'mrkm0', 'mrld0', 'mrlj0',
  'mrlj1', 'mrlk0', 'mrlr0', 'mrmb0', 'mrmg0', 'mrmh0', 'mrml0', 'mrms0', 'mrpc1', 'mrre0',
  'mrso0', 'mrsp0', 'mrtc0', 'mrtj0', 'mrvg0', 'mrwa0', 'mrws0', 'mrxb0', 'msah1', 'msas0',
  'msat0', 'msat1', 'msdb0', 'msdh0', 'msds0', 'msem1', 'mses0', 'msfh0', 'msfv0', 'msjk0',
  'msmc0', 'msmr0', 'msms0', 'msrg0', 'msrr0', 'mstf0', 'msvs0', 'mtab0', 'mtas0', 'mtat0',
  'mtat1', 'mtbc0', 'mtcs0', 'mtdb0', 'mtdp0', 'mter0', 'mtjg0', 'mtjm0', 'mtjs0', 'mtju0',
  'mtkd0', 'mtkp0', 'mtlb0', 'mtlc0', 'mtml0', 'mtmn0', 'mtmt0', 'mtpf0', 'mtpg0', 'mtpp0',
  'mtpr0', 'mtqc0', 'mtrc0', 'mtrr0', 'mtrt0', 'mtwh1', 'mtxs0', 'mvjh0', 'mvlo0', 'mvrw0',
  'mwac0', 'mwad0', 'mwar0', 'mwch0', 'mwdk0', 'mwem0', 'mwgr0', 'mwre0', 'mwrp0', 'mwsb0',
  'mwsh0', 'mzmb0'
]

VAL_SPEAKERS = [
  'fadg0', 'faks0', 'fcal1', 'fcmh0', 'fdac1', 'fdms0', 'fdrw0', 'fedw0', 'fgjd0', 'fjem0',
  'fjmg0', 'fjsj0', 'fkms0', 'fmah0', 'fmml0', 'fnmr0', 'frew0', 'fsem0', 'majc0', 'mbdg0',
  'mbns0', 'mbwm0', 'mcsh0', 'mdlf0', 'mdls0', 'mdvc0', 'mers0', 'mgjf0', 'mglb0', 'mgwt0',
  'mjar0', 'mjfc0', 'mjsw0', 'mmdb1', 'mmdm2', 'mmjr0', 'mmwh0', 'mpdf0', 'mrcs0', 'mreb0',
  'mrjm4', 'mrjr0', 'mroa0', 'mrtk0', 'mrws1', 'mtaa0', 'mtdt0', 'mteb0', 'mthc0', 'mwjg0'
]

TEST_SPEAKERS = [
  'fdhc0', 'felc0', 'fjlm0', 'fmgd0', 'fmld0', 'fnlp0', 'fpas0', 'fpkt0', 'mbpm0', 'mcmj0',
  'mdab0', 'mgrt0', 'mjdh0', 'mjln0', 'mjmp0', 'mklt0', 'mlll0', 'mlnt0', 'mnjm0', 'mpam0',
  'mtas1', 'mtls0', 'mwbt0', 'mwew0'
]

DEFAULT_RAW_DATA_DIR = os.path.expanduser(os.path.join('~', 'Data', 'TIMIT_raw'))
DEFAULT_DATA_DIR = os.path.expanduser(os.path.join('~', 'Data', 'TIMIT'))


def recursive_lowercase_rename(directory):
  """ Convert all directory names and filenames to lowercase.

  Args:
    directory: A string. The top-level directory.
  """
  for dirpath, dirnames, filenames in os.walk(directory, topdown=False):
    for name in dirnames + filenames:
      old_path = os.path.join(dirpath, name)
      new_path = os.path.join(dirpath, name.lower())
      os.rename(old_path, new_path)


def main():
  """ Convert raw TIMIT to the standard train, val, test sets. """

  description = main.__doc__
  formatter_class = argparse.ArgumentDefaultsHelpFormatter
  parser = argparse.ArgumentParser(description=description, formatter_class=formatter_class)
  parser.add_argument('--raw_data_dir', type=str, default=DEFAULT_RAW_DATA_DIR,
                      help='''The raw TIMIT data directory. This should contain the file readme.doc
                              and the directories doc, test, and train. It's okay if they're uppercase,
                              lowercase, etc. In the standardized version, everything will be lowercase.''')
  parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                      help='''The directory in which we'll save the standardized version.''')
  args = parser.parse_args()

  top_level_contents = [name.lower() for name in os.listdir(args.raw_data_dir)]
  if not all(name in top_level_contents for name in ['readme.doc', 'doc', 'test', 'train']):
    raise ValueError('%s is not the expected raw TIMIT directory. Run timit.py --help.' % args.raw_data_dir)

  if os.path.exists(args.data_dir):
    raise ValueError('%s already exists. ' % args.data_dir + 'If you intend to overwrite it, delete it first.')

  raw_train_dir = os.path.join(args.data_dir, 'raw_train')
  raw_test_dir = os.path.join(args.data_dir, 'raw_test')

  train_dir = os.path.join(args.data_dir, 'train')
  val_dir = os.path.join(args.data_dir, 'val')
  test_dir = os.path.join(args.data_dir, 'test')

  print('Copying %s to %s ..' % (args.raw_data_dir, args.data_dir))
  shutil.copytree(args.raw_data_dir, args.data_dir)

  print('Renaming all files to lowercase..')
  recursive_lowercase_rename(args.data_dir)

  print('Moving train to raw_train and test to raw_test..')
  shutil.move(train_dir, raw_train_dir)
  shutil.move(test_dir, raw_test_dir)

  print('Populating the train directory..')
  for path in glob.glob(os.path.join(raw_train_dir, '*', '*')):
    _, speaker = os.path.split(path)
    if speaker in TRAIN_SPEAKERS:
      shutil.copytree(path, os.path.join(train_dir, speaker))

  print('Populating the val directory..')
  for path in glob.glob(os.path.join(raw_test_dir, '*', '*')):
    _, speaker = os.path.split(path)
    if speaker in VAL_SPEAKERS:
      shutil.copytree(path, os.path.join(val_dir, speaker))

  print('Populating the test directory..')
  for path in glob.glob(os.path.join(raw_test_dir, '*', '*')):
    _, speaker = os.path.split(path)
    if speaker in TEST_SPEAKERS:
      shutil.copytree(path, os.path.join(test_dir, speaker))

  for parent_dir in [train_dir, val_dir, test_dir]:
    print('Recursively removing dialect files from %s ..' % parent_dir)
    for dirpath, dirnames, filenames in os.walk(parent_dir):
      for filename in filenames:
        if filename.startswith('sa'):
          os.remove(os.path.join(dirpath, filename))

  for parent_dir in [train_dir, val_dir, test_dir]:
    print('Recursively repairing wav files (by adding headers) in %s ..' % parent_dir)
    for dirpath, dirnames, filenames in os.walk(parent_dir):
      for filename in filenames:
        if filename.endswith('wav'):
          path = os.path.join(dirpath, filename)
          raw_path = os.path.join(dirpath, 'raw_' + filename)
          os.rename(path, raw_path)
          os.system('sox %s %s' % (raw_path, path))
          os.remove(raw_path)


if __name__ == '__main__':
  main()
