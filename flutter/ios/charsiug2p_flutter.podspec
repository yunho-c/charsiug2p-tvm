Pod::Spec.new do |s|
  s.name             = 'charsiug2p_flutter'
  s.version          = '0.1.0'
  s.summary          = 'Flutter Rust Bridge bindings for charsiug2p TVM runtime.'
  s.description      = <<-DESC
Flutter Rust Bridge bindings for charsiug2p TVM runtime.
                       DESC
  s.homepage         = 'https://example.com'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'charsiug2p' => 'noreply@example.com' }
  s.source           = { :path => '.' }
  s.source_files     = 'Classes/**/*'
  s.dependency 'Flutter'
  s.platform         = :ios, '12.0'
  s.swift_version    = '5.0'
end
