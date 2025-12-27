Pod::Spec.new do |s|
  s.name             = 'charsiug2p_flutter'
  s.version          = '0.1.0'
  s.summary          = 'Flutter Rust Bridge bindings for charsiug2p-tvm.'
  s.description      = <<-DESC
Flutter Rust Bridge bindings for charsiug2p-tvm.
                       DESC
  s.homepage         = 'https://github.com/yunho-c/charsiug2p-tvm'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'charsiug2p' => 'opensource@yunhocho.com' }
  s.source           = { :path => '.' }
  s.source_files     = 'Classes/**/*'
  s.dependency 'FlutterMacOS'
  s.dependency 'tvm_flutter'
  s.platform         = :osx, '12.0'
  s.swift_version    = '5.0'
  s.script_phase = {
    :name => 'Build Rust library',
    :script => 'sh "$PODS_TARGET_SRCROOT/../cargokit/build_pod.sh" ../rust charsiug2p_g2p_ffi',
    :execution_position => :before_compile,
    :input_files => ['${BUILT_PRODUCTS_DIR}/cargokit_phony'],
    :output_files => ["${BUILT_PRODUCTS_DIR}/libcharsiug2p_g2p_ffi.a"],
  }
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    # Link system lib containing compiled TVM models
    'OTHER_LDFLAGS' => '-force_load ${BUILT_PRODUCTS_DIR}/libcharsiug2p_g2p_ffi.a ' +
                       '-force_load ${PODS_TARGET_SRCROOT}/../assets/metal-macos/libg2p_system_lib.a',
  }
end

