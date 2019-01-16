# The Native Erlang Guide
This guide includes how to install Erlang from source, with or without HiPE.


## Native Erlang with HiPE

In order to nativly compile Erlang code we need to compile both our own code and
the libraries we use, including Erlang's standard library).
To compile Erlang's standard library with HiPE see the instructions [below](#native) where this is achieved by compiling from source with an extra option.
Note that we can not compile code using NIFs with HiPE because of a built-in [limitation](http://erlang.org/doc/man/HiPE_app.html#feature-limitations).


## Installing Erlang from source
We used Erlang OTP 21 from http://www.erlang.org/download/otp_src_21.0.tar.gz.

### Instructions
Download erlang and go to the source code:
```bash
$ wget http://www.erlang.org/download/otp_src_21.0.tar.gz
$ tar -xf otp_src_21.0.tar.gz
$ cd otp_src_21.0
```

<a name="native"></a>
**Compile Erlang with no options** ([source](http://erlang.org/doc/installation_guide/INSTALL.html)):
``` bash
$ ./configure
$ make
$ sudo make install

```
Without using `--prefix=/my/path` with `./configure` it will use a default path for the installation.
In our case it was `/usr/local/lib/erlang/`.
Also remember to set some environment variables for running C Nodes and NIFs:
```bash
export C_INCLUDE_PATH="/usr/local/lib/erlang/erts-10.0/include/"
export C_INCLUDE_PATH="/usr/local/lib/erlang/lib/erl_interface-3.10.3/include/:$C_INCLUDE_PATH"
export LIBRARY_PATH="/usr/local/lib/erlang/lib/erl_interface-3.10.3/lib/"
```

Alternatively, **Compile Erlang with native libraries** ([source](http://erlang.org/pipermail/erlang-questions/2014-April/078460.html)):
``` bash
$ ./configure --enable-native-libs
$ make clean # you need to run ./configure before clean as you don’t have a Makefile otherwise
$ ./configure --enable-native-libs # as the make clean will have deleted the settings
$ make
$ sudo make install

```
To verify that the libraries are compiled with HiPE run `erl` and check an arbitrary module in Erlang's standard library with `code:is_module_native/1` like this:
```
Eshell V10.0  (abort with ^G)
1> code:is_module_native(lists).
true
```
Note that this function will give false when checking the `erlang` module.
