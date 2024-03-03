## Install
```bash
git clone https://github.com/epignatelli/calf
cd calf
pip install -e .

git clone https://github.com/epignatelli/minihack
cd minihack
pip install -e . --config-settings editable_mode=strict

git clone https://github.com/epignatelli/helx
cd helx
git checkout calf
pip install -e . --config-settings editable_mode=strict
```