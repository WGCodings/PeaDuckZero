# make                 engine binary ./peaduckzero (native CPU), network from EVALFILE
# make release         cargo build --release (engine + duckmatch in target/release)
# make test            all tests
EXE      := peaduckzero
EVALFILE ?= networks/default.pdn
export EVALFILE

ifeq ($(OS),Windows_NT)
	NAME := $(EXE).exe
else
	NAME := $(EXE)
endif

.PHONY: native release test clean

native:
	cargo rustc --release -p peaduckzero --bin peaduckzero -- -C target-cpu=native --emit link=$(NAME)

release:
	cargo build --release

test:
	cargo test --release

clean:
	cargo clean